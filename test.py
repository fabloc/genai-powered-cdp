import pandas_gbq
from pandas_gbq.exceptions import AccessDenied, GenericGBQException
import time
import cfg, re

sql = '''SELECT
      name, gender,
      SUM(number) AS total
    FROM
      `cdp-demo-flocquet.publisher_1_dataset.hll_user_aggregates`
    GROUP BY
      name, gender
    ORDER BY
      total DESC
    LIMIT
      10;'''

def execute_sql(generated_sql):
  """Executes the given SQL query using the pandas_gbq library.

  Args:
    generated_sql: The SQL query to execute.

  Returns:
    A pandas DataFrame containing the results of the query.
  """

  # Set the initial backoff time to 1 second.
  backoff_time = 1

  # Set the maximum number of retries to 5.
  max_retries = 5

  # Create a counter to track the number of retries.
  retry_count = 0

  while retry_count < max_retries:
    try:
      # Execute the SQL query.
      df = pandas_gbq.read_gbq(generated_sql, project_id=cfg.project_id)

      # If the query is successful, return the results.
      return df

    except GenericGBQException as err:
      
      # If the query fails, check the error code.
      error_code_pattern = re.compile('Reason: (\d+)')
      error_message_raw = err.args[0]
      match = error_code_pattern.findall(error_message_raw)

      error_code = int(match[0])
      error_message = error_message_raw.split('\n\nLocation')[0]

      # If the error code is 500 or 503, retry the query.
      if error_code == 500 or error_code == 503:
        # Increase the backoff time.
        backoff_time *= 2

        # Sleep for the backoff time.
        time.sleep(backoff_time)

        # Increment the retry count.
        retry_count += 1

      # If the error code is not 500 or 503, raise the exception.
      elif error_code == 403 or error_code == 409:
        raise Exception({'error_type': 'Permission Error', 'error_message': error_message})
      elif error_code == 400 or error_code == 404:
        raise Exception({'error_type': 'Bad SQL query', 'sql_error': error_message})
      else:
        raise Exception({'error_type': 'Undefined BigQuery Error', 'sql_error': error_message_raw})

  # If the maximum number of retries has been reached, raise an exception.
  raise Exception('Maximum number of retries reached.')

try:
    execute_sql(sql)
except Exception as err:
    print(err)