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
      users.country AS country,
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
    FROM {{ dataset_id }}.users AS users
    -- Below is equivalent to 'LIMIT users_number' (LIMIT does not support variables)
    QUALIFY ROW_NUMBER() OVER() < (SELECT users_number)
  )

  SELECT
    user_id,
    country,
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
  LEFT JOIN `{{ dataset_id }}.products` AS products
  ON session_events.product_id = products.id
  GROUP BY user_id, country, session_id, session_created_at, traffic_source

);


-----------------------------------------------------------------------------------------------
-- Create first aggregated table, aggregated at a daily scope
CREATE OR REPLACE TABLE `{{ dataset_id }}.user_aggregates_daily`
PARTITION BY
  DATE(session_day)
AS (

WITH
-- Find the number of items purchased by a customer
-- Find total money spent by a customer for all items
-- Find all product categories purchased by the customer, and how many items/spend for each one TODO
users_purchases AS (
  SELECT
    user_id,
    DATETIME_TRUNC(session_created_at, DAY) AS session_day,
    session_flat.cost AS item_cost,
    session_flat.brand AS brand_name,
    session_flat.product_id,
    session_flat.category AS category
  FROM `{{ dataset_id }}.events` AS events,
  UNNEST(session) AS session_flat
  WHERE total_cost > 0 and cost > 0
),

daily_users_products AS (
  SELECT
    user_id,
    session_day,
    STRUCT(
      brand_name,
      category,
      product_ids,
      purchased_items,
      brand_spend
    ) brands
  FROM (
    SELECT
      user_id,
      session_day,
      brand_name,
      STRING_AGG(DISTINCT category) as category,
      ARRAY_AGG(DISTINCT product_id) AS product_ids,
      COUNT(1) AS purchased_items,
      SUM(item_cost) AS brand_spend
    FROM users_purchases
    GROUP BY user_id, session_day, brand_name
  )
),

daily_user_orders AS (
  SELECT
    users_purchases.session_day,
    users_purchases.user_id,
    daily_purchased_items,
    daily_spend,
    ARRAY_AGG(
      daily_users_products.brands
    ) daily_purchased_products_by_brands
  FROM (
    SELECT
      user_id,
      session_day,
      COUNT(item_cost) AS daily_purchased_items,
      SUM(item_cost) AS daily_spend
    FROM users_purchases
    GROUP BY user_id, session_day
  ) AS users_purchases
  JOIN daily_users_products
  ON users_purchases.user_id = daily_users_products.user_id
  WHERE users_purchases.session_day = daily_users_products.session_day
  GROUP BY users_purchases.user_id, users_purchases.session_day, daily_purchased_items, daily_spend
)

SELECT
  id,
  session_day,
  daily_purchased_items,
  daily_spend,
  daily_purchased_products_by_brands
FROM `{{ dataset_id }}.users` as users
LEFT JOIN daily_user_orders ON users.id = daily_user_orders.user_id

);


-----------------------------------------------------------------------------------------------
-- Create Global Aggregated Table
CREATE OR REPLACE TABLE `{{ dataset_id }}.user_aggregates_global` AS

WITH

-- Find the number of items purchased by a customer
-- Find total money spent by a customer for all items
-- Find all product categories purchased by the customer, and how many items/spend for each one TODO
users_purchases AS (
  SELECT
    user_id,
    total_cost,
    session_created_at,
    session_flat.cost AS item_cost,
    session_flat.product_id,
    session_flat.brand AS brand_name,
    session_flat.category AS category
  FROM `{{ dataset_id }}.events` AS events,
  UNNEST(session) AS session_flat
  WHERE total_cost > 0 and cost > 0
),

global_users_products AS (
  SELECT
    user_id,
    STRUCT(
      brand_name,
      category,
      purchased_items,
      product_ids,
      brand_spend
    ) brands
  FROM (
    SELECT
      user_id,
      brand_name,
      STRING_AGG(DISTINCT category) as category,
      ARRAY_AGG(DISTINCT product_id) AS product_ids,
      COUNT(1) AS purchased_items,
      SUM(item_cost) AS brand_spend
    FROM users_purchases
    GROUP BY user_id, brand_name
  )
),

global_user_orders AS (
  SELECT
    users_purchases.user_id,
    total_purchased_items,
    total_spend,
    first_purchase_date,
    last_purchase_date,
    ARRAY_AGG(
      global_users_products.brands
    ) total_purchased_products_by_brands
  FROM (
    SELECT
      user_id,
      COUNT(item_cost) AS total_purchased_items,
      SUM(item_cost) AS total_spend,
      MIN(session_created_at) first_purchase_date,
      MAX(session_created_at) last_purchase_date
    FROM users_purchases
    GROUP BY user_id
  ) AS users_purchases
  JOIN global_users_products
  ON users_purchases.user_id = global_users_products.user_id
  GROUP BY users_purchases.user_id, total_purchased_items, total_spend, first_purchase_date, last_purchase_date
),

-- Find the latest session with products added to cart
users_with_cart AS
(
  SELECT
    session_id,
    user_id,
    MAX(session_created_at) session_created_at
  FROM `{{ dataset_id }}.events` 
  WHERE 'cart' IN (SELECT event_type FROM UNNEST(session))
  GROUP BY user_id, session_id
),

-- Find the latest session with purchase
users_with_purchase AS
(
  SELECT
    user_id,
    MAX(session_created_at) session_created_at
  FROM `{{ dataset_id }}.events`
  WHERE 'purchase' IN (SELECT event_type FROM UNNEST(session))
  GROUP BY user_id
),

-- Identify users with items added to carts done after the latest purchase
users_with_abandoned_cart AS
(
  SELECT
    users_with_cart.user_id,
    users_with_cart.session_id
  FROM users_with_cart
  LEFT JOIN users_with_purchase ON users_with_cart.user_id = users_with_purchase.user_id
  WHERE users_with_cart.session_created_at > users_with_purchase.session_created_at
)

SELECT
  id,
  age,
  gender,
  country,
  total_purchased_items,
  total_spend,
  first_purchase_date,
  last_purchase_date,
  total_purchased_products_by_brands,
  IF(IFNULL(users_with_abandoned_cart.session_id, 'NULL') != 'NULL', true, false) AS has_abandoned_cart
FROM `{{ dataset_id }}.users` as users
LEFT JOIN global_user_orders ON users.id = global_user_orders.user_id
LEFT JOIN users_with_abandoned_cart ON users.id = users_with_abandoned_cart.user_id;


-----------------------------------------------------------------------------------------------
-- Append to the user_aggregates table based on the daily and global aggregated tables above
INSERT `{{ dataset_id }}.user_aggregates`
SELECT
  global.id AS user_id,
  age,
  gender,
  country,
  total_purchased_items,
  total_spend,
  first_purchase_date,
  last_purchase_date,
  session_day,
  daily_purchased_items,
  daily_spend,
  ARRAY_CONCAT_AGG(daily_purchased_products_by_brands) AS daily_purchased_products_by_brands,
  ARRAY_CONCAT_AGG(total_purchased_products_by_brands) AS total_purchased_products_by_brands,
  has_abandoned_cart
FROM `{{ dataset_id }}.user_aggregates_global` AS global
JOIN `{{ dataset_id }}.user_aggregates_daily` AS daily
ON global.id = daily.id
GROUP BY
  user_id,
  age,
  gender,
  country,
  total_purchased_items,
  total_spend,
  first_purchase_date,
  last_purchase_date,
  session_day,
  daily_spend,
  daily_purchased_items,
  has_abandoned_cart;


-----------------------------------------------------------------------------------------------
-- Create an intermediate aggregated product table, with daily aggregation granularity
CREATE OR REPLACE TABLE `{{ dataset_id }}.product_aggregates_daily`
PARTITION BY
  DATE(session_day)
AS (

  SELECT
    session_flat.product_id AS product_id,
    session_flat.name AS product_name,
    session_flat.brand AS brand_name,
    session_flat.category AS category,
    DATETIME_TRUNC(session_created_at, DAY) AS session_day,
    events.country AS country,
    events.traffic_source AS traffic_source,
    COUNT(session_flat.cost) AS daily_product_purchased_items,
    SUM(CAST(session_flat.cost AS NUMERIC)) AS daily_product_spend
  FROM `{{ dataset_id }}.events` AS events,
  UNNEST(session) AS session_flat
  WHERE cost > 0
  GROUP BY product_id, product_name, brand_name, category, session_day, country, traffic_source
);


-----------------------------------------------------------------------------------------------
-- Create an intermediate aggregated product table, providing global aggregated columns
CREATE OR REPLACE TABLE `{{ dataset_id }}.product_aggregates_global` AS
SELECT
     session_flat.product_id AS product_id,
     session_flat.name AS product_name,
     session_flat.brand AS brand_name,
     session_flat.category AS category,
     COUNT(1) AS product_total_purchased_items,
     SUM(CAST(session_flat.cost AS NUMERIC)) AS product_total_spend,
     MAX(session_created_at) last_product_purchase_date
FROM `{{ dataset_id }}.events` AS events,
UNNEST(session) AS session_flat
WHERE product_id != 0
GROUP BY product_id, product_name, brand_name, category; 


-----------------------------------------------------------------------------------------------
-- Create final aggregated products table based on the daily and global aggregated tables above
INSERT `{{ dataset_id }}.product_aggregates`
SELECT
  global.product_id AS product_id,
  global.product_name AS product_name,
  global.brand_name AS brand_name,
  global.category AS category,
  product_total_purchased_items,
  product_total_spend,
  last_product_purchase_date,
  session_day,
  daily_product_purchased_items,
  daily_product_spend,
  country,
  traffic_source
FROM `{{ dataset_id }}.product_aggregates_global` AS global
JOIN `{{ dataset_id }}.product_aggregates_daily` AS daily
ON global.product_id = daily.product_id