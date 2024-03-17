-- Create a variable containing the total number of products
DECLARE total_products_number DEFAULT (SELECT COUNT(DISTINCT id) FROM {{ dataset_id }}.products);

-- Create a variable containing the number of users to use for events generation
DECLARE users_number DEFAULT (SELECT {{ users_percentage }} * COUNT(DISTINCT id) / 100 FROM {{ dataset_id }}.users);

CREATE TEMP FUNCTION rand_date(min_date DATE,max_date DATE) AS ( 
  TIMESTAMP_SECONDS(
    CAST(
      ROUND(UNIX_SECONDS(CAST(min_date AS TIMESTAMP)) + rand() * (UNIX_SECONDS(CAST(max_date AS TIMESTAMP)) - UNIX_SECONDS(CAST(min_date AS TIMESTAMP))))
    AS INT64)
  ) 
);

CREATE TEMP FUNCTION generate_event_types(session_creation_date TIMESTAMP, total_products INT64)
RETURNS ARRAY<STRUCT<event_type STRING, product_id INT64, created_at TIMESTAMP, sequence_number INT64>>
LANGUAGE js
AS r"""

  function randomDate(start, end) {
    var shiftDate = new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime()));
    return shiftDate;
  }

  const max_session_time = 30

  session = [];
  total_products = total_products - 1;

  // Randomly determine the length of eventTypes between 0 and 6
  length = Math.floor(Math.random() * 7);

  current_created_at = new Date(session_creation_date);
  max_created_at = new Date(new Date(current_created_at).getTime() + max_session_time * 60000);

  // Ensure session is at least 2 if it's not 0
  if( length != 0 ) {

    // Generate the first element of session
    i = 0;
    if (Math.random() <= 0.5) {
      current_created_at = randomDate(current_created_at, max_created_at);
      session.push({created_at: current_created_at, event_type: 'home', product_id: 0, sequence_number: i+1});
      i++;
    }

    // Generate the remaining elements of session
    while (i < length-1) {
      if (Math.random() < 0.5) {
        product_id = Math.floor(Math.random() * total_products) + 1;
        current_created_at = randomDate(current_created_at, max_created_at);
        session.push({created_at: current_created_at, event_type: 'department', product_id: product_id, sequence_number: i+1});
        current_created_at = randomDate(current_created_at, max_created_at);
        session.push({created_at: current_created_at, event_type: 'product', product_id: product_id, sequence_number: i+2});
        i += 2;
      } else {
        current_created_at = randomDate(current_created_at, max_created_at);
        session.push({created_at: current_created_at, event_type: 'product', product_id: Math.floor(Math.random() * total_products) + 1, sequence_number: i+1});
        i++;
      }
    }

    choice = Math.random();
    if(length > 2) {
      session[length - 2] = choice < 0.5 ? {created_at: current_created_at, event_type: 'cart', product_id: 0, sequence_number: length - 1} : session[length - 2];
    }

    if (length > 1 && session[length - 2].event_type == 'cart') {
      current_created_at = randomDate(current_created_at, max_created_at);
      choice = Math.random();
      session[length - 1] = {created_at: current_created_at, event_type: choice < 0.5 ? 'cancel' : 'purchase', product_id: 0, sequence_number: length};
      i++;
    }

    if (i < length) {
      current_created_at = randomDate(current_created_at, max_created_at);
      session.push({created_at: current_created_at, event_type: 'product', product_id: Math.floor(Math.random() * total_products) + 1, sequence_number: i+1});
    }
  }
  return session;
""";

CREATE OR REPLACE TABLE {{ dataset_id }}.events
PARTITION BY
  DATE_TRUNC(session_created_at, DAY)
AS (

  WITH

  traffic_sources AS (
      SELECT ["Search", "Organic", "Email", "Facebook", "Display"] as sources_list
  ),

  products_by_users AS (
    SELECT
      users.id AS user_id,
      -- Each session is a Struct of attribute, among which an Array containing the events for the session, generated using a javascript script
      ARRAY(
        -- Use a SubQuery to precompute the session creation date and reuse it as input to the session events generation UDF
        SELECT AS STRUCT
          session_id,
          traffic_source,
          session_created_at,
          generate_event_types(session_created_at, total_products_number) AS session
        FROM (
          SELECT
            (SELECT sources_list[mod(CAST(5*rand() as int64), 5)] FROM traffic_sources) traffic_source,
            CAST(rand_date("{{ events_start_date }}", "{{ events_end_date }}") AS TIMESTAMP) session_created_at,
            GENERATE_UUID() AS session_id
          -- Create an array of a random number of sessions (between 0 and a user defined variable) using an Unnest trick to loop through the sessions.
          FROM UNNEST(GENERATE_ARRAY(1, CAST({{ max_sessions_per_user }} * RAND() AS INT64)))
        )
      ) AS sessions
    FROM {{ dataset_id }}.users AS users_number
    -- Below is equivalent to 'LIMIT users_number' (LIMIT does not support variables)
    QUALIFY ROW_NUMBER() OVER() < (SELECT users_number)
  )

  SELECT
    user_id,
    session_id,
    traffic_source,
    flat_sessions.session_created_at AS session_created_at,
    SUM(
      CASE
        -- Check whether 'event_type' has value 'purchase' in the session events. If yes, add the costs of all session rows
        WHEN EXISTS(SELECT * FROM flat_sessions.session AS session WHERE event_type = 'purchase') THEN
          cost
        ELSE
          0
      END
    ) AS total_cost,
    ARRAY_AGG(STRUCT(
      session_events.event_type AS event_type,
      session_events.product_id AS product_id,
      CAST(session_events.created_at AS TIMESTAMP) AS created_at,
      session_events.sequence_number AS sequence_number,
      (CASE
        WHEN session_events.event_type = 'department' THEN
          CONCAT('/department/', REPLACE(LOWER(department), ' ', ''), '/category/', REPLACE(LOWER(category), ' ', ''), '/brand/', REPLACE(LOWER(brand), ' ', ''))
        WHEN session_events.event_type = 'product' THEN
          CONCAT('/product/', product_id)
      END) AS uri,
      cost AS cost,
      category AS category,
      name AS name,
      brand AS brand,
      department AS department
      )) AS session
  FROM
    products_by_users,
    -- Unnest the array of session 'sessions', and then unnest the events inside the sessions
    UNNEST(sessions) AS flat_sessions,
    UNNEST(flat_sessions.session) AS session_events
  LEFT JOIN {{ dataset_id }}.products AS products
  ON session_events.product_id = products.id
  GROUP BY user_id, session_id, session_created_at, traffic_source

)