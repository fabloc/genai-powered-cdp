
from google.cloud import bigquery
import pandas_gbq, time, logging, re
from datetime import datetime
import pandas as pd
from google.cloud.exceptions import NotFound
import cfg


# Construct a BigQuery client object.
bq_client = bigquery.Client(location=cfg.dataset_location, project=cfg.dataproject_id)

job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

# create logger
global logger
logger = logging.getLogger('pgvector_handler')


# Enable NL2SQL Analytics Warehouse
if cfg.enable_analytics is True:

    # Create a dataset
    try:
      dataset = bq_client.create_dataset(dataset=cfg.dataset_name)
    except Exception as e:
      logger.error('Failed to create the dataset\n')
      logger.error(str(e))

def execute_bq_query(sql_query, dry_run: bool = True):

  # Set the initial backoff time to 1 second.
  backoff_time = 1

  # Set the maximum number of retries to 5.
  max_retries = 5

  # Create a counter to track the number of retries.
  retry_count = 0

  bq_status = {
    'status': 'Success',
    'error_type': None,
    'error_code': None,
    'error_message': None,
    'execution_time': -1
  }

  while retry_count < max_retries:

    df = None

    try:

      if dry_run is True:
        # Start the query, passing in the extra configuration.
        query_job = bq_client.query(
          (sql_query),
          job_config=job_config,
        )  # Make an API request.

        # A dry run query completes immediately.
        print("This query will process {} bytes.".format(query_job.total_bytes_processed))
      
      else:
        # Execute the SQL query.
        df = pandas_gbq.read_gbq(sql_query, project_id=cfg.project_id)
    
      # If the query is successful, return the results.
      bq_status['status'] = 'Success'
      logger.info('Query execution success...')
      return bq_status,df
    
    except Exception as e:
      # If the query fails, check the error code.
      if dry_run is True:
        error_code = e.response.status_code
        error_message = e.errors[0]['message']
      else:
        # If the query fails, check the error code.
        error_code_pattern = re.compile('Reason: (\d+)')
        error_message_raw = e.args[0]
        match = error_code_pattern.findall(error_message_raw)

        error_code = int(match[0])
        error_message = error_message_raw.split('\n\nLocation')[0]

      logger.info("BigQuery execution failed with error code: " + str(error_code) + " and error message: " + error_message)

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
        bq_status['status'] = 'Error'
        bq_status['error_type'] = 'Permission Error'
        bq_status['error_code'] = error_code
        bq_status['error_message'] = error_message
        return bq_status,None
      elif error_code == 400 or error_code == 404:
        bq_status['status'] = 'Error'
        bq_status['error_type'] = 'Bad SQL Query'
        bq_status['error_code'] = error_code
        bq_status['error_message'] = error_message
        return bq_status, None
      else:
        bq_status['status'] = 'Error'
        bq_status['error_type'] = 'Undefined BigQuery Error'
        bq_status['error_code'] = error_code
        bq_status['error_message'] = error_message
        return bq_status, None

  # If the query is still failing after the maximum number of retries, raise an exception.
  bq_status['status'] = 'Error'
  bq_status['error_type'] = 'BigQuery Internal Error'
  bq_status['error_message'] = 'Maximum number of retries reached.'
  return bq_status, None

def append_2_bq(model, question, generated_sql, found_in_vector, need_rewrite, failure_step, error_msg):

  if cfg.enable_analytics is True:
      logger.debug('\nInside the Append to BQ block\n')
      table_id=cfg.project_id + '.' + cfg.dataset_name + '.' + cfg.log_table_name
      now = datetime.now()

      table_exists=False
      client = bigquery.Client()

      df1 = pd.DataFrame(columns=[
          'source_type',
          'project_id',
          'user',
          'schema',
          'model_used',
          'question',
          'generated_sql',
          'found_in_vector',
          'need_rewrite',
          'failure_step',
          'error_msg',
          'execution_time'
          ])

      new_row = {
          "source_type":cfg.source_type,
          "project_id":str(cfg.project_id),
          "user":str(cfg.auth_user),
          "schema": cfg.schema,
          "model_used": model,
          "question": question,
          "generated_sql": generated_sql,
          "found_in_vector":found_in_vector,
          "need_rewrite":need_rewrite,
          "failure_step":failure_step,
          "error_msg":error_msg,
          "execution_time": now
        }

      df1.loc[len(df1)] = new_row

      db_schema=[
            # Specify the type of columns whose type cannot be auto-detected. For
            # example the "title" column uses pandas dtype "object", so its
            # data type is ambiguous.
            bigquery.SchemaField("source_type", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("project_id", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("user", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("schema", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("model_used", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("question", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("generated_sql", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("found_in_vector", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("need_rewrite", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("failure_step", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("error_msg", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("execution_time", bigquery.enums.SqlTypeNames.TIMESTAMP),
            bigquery.SchemaField("full_log", bigquery.enums.SqlTypeNames.STRING),
          ]

      try:
        client.get_table(table_id)  # Make an API request.
        #llogger.info("Table {} already exists.".format(table_id))
        table_exists=True
      except NotFound:
          logger.error("Table {} is not found.".format(table_id))
          table_exists=False

      if table_exists is True:
          logger.info('Performing streaming insert')
          errors = client.insert_rows_from_dataframe(table=table_id, dataframe=df1, selected_fields=db_schema)  # Make an API request.
          #if errors == []:
          #    print("New rows have been added.")
          #else:
          #    print("Encountered errors while inserting rows: {}".format(errors))
      else:
          pandas_gbq.to_gbq(df1, table_id, project_id=cfg.project_id)  # replace to replace table; append to append to a table


      # df1.loc[len(df1)] = new_row
      # pandas_gbq.to_gbq(df1, table_id, project_id=PROJECT_ID, if_exists='append')  # replace to replace table; append to append to a table
      logger.info('Query added to BQ log table')
      return 'Row added'
  else:
    logger.debug('BQ Analytics is disabled so query was not added to BQ log table')

    return 'BQ Analytics is disabled'