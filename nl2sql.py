# Common Imports
import pandas_gbq
import pgvector_handler
import os, json, time, yaml, sys
import logging.config
import cfg
import genai
import bigquery_handler
from streamlit.elements.lib.mutable_status_container import StatusContainer
from concurrent.futures import ThreadPoolExecutor

# Define BigQuery Dictionary Queries and Helper Functions

get_columns_sql='''
SELECT
    columns.TABLE_CATALOG as project_id, columns.TABLE_SCHEMA as owner , columns.TABLE_NAME as table_name, columns_field_paths.FIELD_PATH as column_name,
    columns.IS_NULLABLE as is_nullable, columns_field_paths.DATA_TYPE as data_type, columns.COLUMN_DEFAULT as column_default, columns.ROUNDING_MODE as rounding_mode, DESCRIPTION as column_description
  FROM
    {cfg.user_dataset}.INFORMATION_SCHEMA.COLUMNS AS columns
  JOIN
    {cfg.user_dataset}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS AS columns_field_paths
  ON columns.TABLE_CATALOG = columns_field_paths.TABLE_CATALOG AND columns.TABLE_SCHEMA = columns_field_paths.TABLE_SCHEMA
    AND columns.TABLE_NAME = columns_field_paths.TABLE_NAME AND columns.COLUMN_NAME = columns_field_paths.COLUMN_NAME
  WHERE
    CASE
        WHEN ARRAY_LENGTH({cfg.tables}) > 0 THEN columns.table_name IN UNNEST({cfg.tables})
        ELSE TRUE
    END
  ORDER BY
  project_id, owner, columns.table_name, columns.column_name ;
'''


get_fkeys_sql='''
SELECT T.CONSTRAINT_CATALOG, T.CONSTRAINT_SCHEMA, T.CONSTRAINT_NAME,
T.TABLE_CATALOG as project_id, T.TABLE_SCHEMA as owner, T.TABLE_NAME as table_name, T.CONSTRAINT_TYPE,
T.IS_DEFERRABLE, T.ENFORCED, K.COLUMN_NAME
FROM
{cfg.user_dataset}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS T
JOIN {cfg.user_dataset}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE K
ON K.CONSTRAINT_NAME=T.CONSTRAINT_NAME
WHERE
T.CONSTRAINT_TYPE="FOREIGN KEY" AND 
(CASE
    WHEN ARRAY_LENGTH({cfg.tables}) > 0 THEN T.table_name IN UNNEST({cfg.tables})
    ELSE TRUE END)
ORDER BY
project_id, owner, table_name
'''

get_pkeys_sql='''
SELECT T.CONSTRAINT_CATALOG, T.CONSTRAINT_SCHEMA, T.CONSTRAINT_NAME,
T.TABLE_CATALOG as project_id, T.TABLE_SCHEMA as owner, T.TABLE_NAME as table_name, T.CONSTRAINT_TYPE,
T.IS_DEFERRABLE, T.ENFORCED, K.COLUMN_NAME
FROM
    {cfg.user_dataset}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS T
JOIN {cfg.user_dataset}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE K
ON K.CONSTRAINT_NAME=T.CONSTRAINT_NAME
WHERE
    T.CONSTRAINT_TYPE="PRIMARY KEY" AND
    (CASE
        WHEN ARRAY_LENGTH({cfg.tables}) > 0 THEN T.table_name IN UNNEST({cfg.tables})
        ELSE TRUE END)
ORDER BY
project_id, owner, table_name
'''


get_table_comments_sql='''
select TABLE_CATALOG as project_id, TABLE_SCHEMA as owner, TABLE_NAME as table_name, OPTION_NAME, OPTION_TYPE, TRIM(OPTION_VALUE, '"') as comments
FROM
    {cfg.user_dataset}.INFORMATION_SCHEMA.TABLE_OPTIONS
WHERE
    OPTION_NAME = "description" AND
    (CASE
        WHEN ARRAY_LENGTH({cfg.tables}) > 0 THEN table_name IN UNNEST({cfg.tables})
        ELSE TRUE END)
ORDER BY
project_id, owner, table_name
'''

executor = ThreadPoolExecutor(5)


def schema_generator(sql):
  formatted_sql = sql.format(**globals(), **locals())
  logger.info("BigQuery request: " + formatted_sql)
  df = pandas_gbq.read_gbq(formatted_sql, project_id=cfg.project_id, location=cfg.region)
  return df


def add_table_comments(columns_df, pkeys_df, fkeys_df, table_comments_df):

  for index, row in table_comments_df.iterrows():
    if row['comments'] is None: ## or row['comments'] is not None:
        context_prompt = f"""
        Generate table comments for the table {row['project_id']}.{row['owner']}.{row['table_name']}

        Parameters:
        - column metadata: {columns_df.to_markdown(index = False)}
        - primary key metadata: {pkeys_df.to_markdown(index = False)}
        - foreign keys metadata: {fkeys_df.to_markdown(index = False)}
        - table metadata: {table_comments_df.to_markdown(index = False)}
      """
        context_query = genai.generate_sql(context_prompt)
        table_comments_df.at[index, 'comments'] = context_query

  return table_comments_df


def get_tables(df):
  tables = []
  for _, row in df.iterrows():
      tables.append(row['table_name'])
  df.reset_index()
  return tables

def insert_sample_queries_lookup(tables_list):
  queries_samples = []
  for table_name in tables_list:
    samples_filename = 'queries_samples/' + table_name + '.yaml'
    if os.path.exists(samples_filename):
      with open(samples_filename) as stream:
        try:
          table_queries_samples = yaml.safe_load(stream)
          queries_samples.append(table_queries_samples)
          for sql_query in queries_samples[0]:
            question = sql_query['Question']
            question_text_embedding = pgvector_handler.text_embedding(question)
            pgvector_handler.add_vector_sql_collection(cfg.schema, sql_query['Question'], sql_query['SQL Query'], question_text_embedding, 'Y')
        except yaml.YAMLError as exc:
          logger.error("Error loading YAML file containing sample questions: " + exc)
          logger.error("Skipping sample ingestion")
        except Exception as err:
          logger.error("Error: " + err.error_message())
        finally:
          stream.close()

  return queries_samples


def init_table_and_columns_desc():
  
    table_comments_df= schema_generator(get_table_comments_sql)

    # List all tables to be considered
    tables_list = get_tables(table_comments_df)

    # Test whether all tables descriptions are present in pgvector DB
    # If not, generate them
    uninitialized_tables = pgvector_handler.pgvector_table_desc_exists(table_comments_df)
    if len(uninitialized_tables) == 0:
      logger.info("All defined or identified tables are already present in pgVector DB")
    else:
      logger.info("At least one table is not initialized in pgVector")

      # Store the table, column definition, primary/foreign keys and comments into Dataframes
      # Use Global variable TABLES, updated with uninitialized tables only. A better way needs to be implemented
      global TABLES
      TABLES = uninitialized_tables
      columns_df = schema_generator(get_columns_sql)
      fkeys_df = schema_generator(get_fkeys_sql)
      pkeys_df = schema_generator(get_pkeys_sql)

      # Adding column sample_values to the columns dataframe
      #columns_df = get_column_sample(columns_df)

      # Look at each tables dataframe row and use LLM to generate a table comment, but only for the tables with null comments (DB did not have comments on table)
      ## Using Palm to add table comments if comments are null
      table_comments_df = add_table_comments(columns_df, pkeys_df, fkeys_df, table_comments_df)

      table_comments_df = build_table_desc(table_comments_df,columns_df,pkeys_df,fkeys_df)

      pgvector_handler.add_table_desc_2_pgvector(table_comments_df)

    # Look for files listing sample queries to be ingested in the pgVector DB
    # insert_sample_queries_lookup(tables_list)


# Build a custom "detailed_description" table column to be indexed by the Vector DB
# Augment Table dataframe with detailed description. This detailed description column will be the one used as the document when adding the record to the VectorDB
def build_table_desc(table_comments_df,columns_df,pkeys_df,fkeys_df):
  aug_table_comments_df = table_comments_df

  #self.logger.info(len(aug_table_comments_df))
  #self.logger.info(len(table_comments_df))

  cur_table_name = ""
  cur_table_owner = ""
  cur_project_id = ""
  cur_full_table= cur_project_id + '.' + cur_table_owner + '.' + cur_table_name

  for index_aug, row_aug in aug_table_comments_df.iterrows():

    cur_table_name = str(row_aug['table_name'])
    cur_table_owner = str(row_aug['owner'])
    cur_project_id = str(row_aug['project_id'])
    cur_full_table= cur_project_id + '.' + cur_table_owner + '.' + cur_table_name
    #self.logger.info('\n' + cur_table_owner + '.' + cur_table_name + ':')

    table_cols=[]
    table_pk_cols=[]
    table_fk_cols=[]

    for index, row in columns_df.loc[ (columns_df['owner'] == cur_table_owner) & (columns_df['table_name'] == cur_table_name) ].iterrows():
      # Inside each owner.table_name combination
      table_cols.append('- ' + row['column_name'] + ' (' + row['data_type'] + ') - ' + row['column_description'])

    for index, row in pkeys_df.loc[ (pkeys_df['owner'] == cur_table_owner) & (pkeys_df['table_name'] == cur_table_name)  ].iterrows():
      # Inside each owner.table_name combination
      table_pk_cols.append( row['COLUMN_NAME']  )

    for index, row in fkeys_df.loc[ (fkeys_df['owner'] == cur_table_owner) & (fkeys_df['table_name'] == cur_table_name) ].iterrows():
      # Inside each owner.table_name combination
      fk_cols_text=f"""
      Column {row['column_name']} is equal to column {row['r_column_name']} in table {row['owner']}.{row['r_table_name']}
      """
      table_fk_cols.append(fk_cols_text)


    if len(",".join(table_pk_cols)) == 0:
      final_pk_cols = "None"
    else:
      final_pk_cols = ",".join(table_pk_cols)

    if len(",".join(table_fk_cols)) == 0:
      final_fk_cols = "None"
    else:
      final_fk_cols = ",".join(table_fk_cols)

    ln = ' |\n    '
    aug_table_desc=f"""
    Table: `{cur_full_table}`
    Owner: {cur_table_owner}
    Column (type) - Description - Scope:
    {ln.join(table_cols)}
    Primary Key: {final_pk_cols}
    Foreign Keys: {final_fk_cols}
    Project_id: {str(row_aug['project_id'])}
    Table Description: {str(row_aug['comments'])}
    """

    # Works well
    aug_table_comments_df.at[index_aug, 'detailed_description'] = aug_table_desc
    logger.info("Table schema: \n" + aug_table_desc)
  return aug_table_comments_df


# Define a function to execute other functions with blocking I/Os in a thread with a timeout, in order to handle cases where the
# the I/O requests get stuck forever.
def execute_with_timeout(func, *args, **kwargs):
  # Set the initial backoff time to 1 second.
  backoff_time = 1

  # Set the maximum number of retries to 5.
  max_retries = 5

  # Create a counter to track the number of retries.
  retry_count = 0

  timeout = cfg.models_timeout

  while retry_count < max_retries:
    future_sql_validation = executor.submit(func, *args, **kwargs)
    try:
      response = future_sql_validation.result(timeout)
      return response
    except TimeoutError:
      logger.error('Time out waiting for model response. Retrying...')

      # Increase the backoff time.
      backoff_time *= 2

      # Sleep for the backoff time.
      time.sleep(backoff_time)

      # Increment the retry count.
      retry_count += 1
  
  logger.critical('Max retries exceeded.')
  raise TimeoutError


def call_gen_sql(question, streamlit_status: StatusContainer):

  total_start_time = time.time()
  generated_sql = ''
  sql_result_df = None
  matched_tables = []

  metrics = {
    'embedding_duration': -1,
    'similar_questions_duration': -1,
    'table_matching_duration': -1,
    'sql_generation_duration': -1,
    'bq_validation_duration': -1,
    'bq_execution_duration': -1,
    'sql_added_to_vector_db_duration': -1
  }

  workflow = {
    'stop_loop': False,
    'error_retry_max_count': cfg.sql_max_error_retry,
    'error_retry_count': 0,
    'explanation_retry_count': 0,
    'first_loop': True
  }

  status = {
    'status': 'Error',
    'unrelated_question': False,
    'sql_generation_success': False,
    'sql_validation_success': False,
    'error_messages': None
  }

  streamlit_status.write("Generating SQL Request")

  logger.info("Creating text embedding from question...")
  start_time = time.time()
  question_text_embedding = pgvector_handler.text_embedding(question)
  metrics['embedding_duration'] = time.time() - start_time
  logger.info("Text embedding created")

  logger.info("User questions: " + str(question))

  # Will look into the Vector DB first and see if there is a hash match.
  # If yes, return the known good SQL.
  # If not, return 3 good examples to be used by the LLM
  #search_sql_vector_by_id_return = pgvector_handler.search_sql_vector_by_id(schema, question,'Y')
  logger.info("Look for exact same question in pgVector...")
  search_sql_vector_by_id_return = 'SQL Not Found in Vector DB'

  # search_sql_vector_by_id_return = "SQL Not Found in Vector DB"

  if search_sql_vector_by_id_return == 'SQL Not Found in Vector DB':   ### Only go thru the loop if hash of the question is not found in Vector.

    logger.info("Did not find same question in DB")
    logger.info("Searching for similar questions in DB...")
    start_time = time.time()
    # Look into Vector for similar queries. Similar queries will be added to the LLM prompt (few shot examples)
    similar_questions = pgvector_handler.search_sql_nearest_vector(cfg.schema, question, question_text_embedding, 'Y')

    metrics['similar_questions_duration'] = time.time() - start_time
    logger.info("Found similar questions:\n" + str(similar_questions))

    error_correction_chat_session=None
    logger.info("Now looking for appropriate tables in Vector to answer the question...")
    start_time = time.time()
    table_result_joined = pgvector_handler.get_tables_colums_vector(question, question_text_embedding)
    metrics['table_matching_duration'] = time.time() - start_time

    if len(table_result_joined) > 0 :
        logger.info("Found matching tables")

        # Add matched table to list of tables used during the SQL generation
        for table in cfg.tables:
          if table in table_result_joined:
            matched_tables.append(table)

        start_time = time.time()
        logger.info("Generating SQL query using LLM...")
        generated_sql = genai.gen_dyn_rag_sql(question,table_result_joined, similar_questions)
        streamlit_status.write("SQL Query Generated")
        logger.info("SQL query generated:\n" + generated_sql)
        metrics['sql_generation_duration'] = time.time() - start_time
        if 'unrelated_answer' in generated_sql :
          workflow['stop_loop']=True
          #logger.info('Inside if statement to check for unrelated question...')
          workflow['unrelated_question']=True
        if cfg.inject_one_error is True:
          if workflow['error_retry_count'] < 1:
            logger.info('Injecting error on purpose to test code ... Adding ROWID at the end of the string')
            generated_sql=generated_sql + ' ROWID'
    else:
        workflow['stop_loop']=True
        workflow['unrelated_question']=True
        status['sql_generation_success'] = False
        logger.info('No ANN/appropriate tables found in Vector to answer the question. Stopping...')

    while workflow['stop_loop'] is False:

      # If similar requests were found, Execute SQL test plan and semantic validation in parallel in order to minimize
      # overall query generation time, as there is a high probability that the correct SQL Query will be found the first iteration.
      logger.info("Executing SQL test plan and SQL query semantic validation in parallel...")

      future_test_plan = executor.submit(bigquery_handler.execute_bq_query, generated_sql, dry_run=True)
      future_sql_validation = executor.submit(
        execute_with_timeout,
        genai.sql_explain,
        question,
        generated_sql,
        table_result_joined,
        similar_questions)
      status['bq_status'],_ = future_test_plan.result()
      streamlit_status.write("SQL Query Syntax Check: " + status['bq_status']['status'])
      sql_explanation = future_sql_validation.result()
      streamlit_status.write("SQL Query Matches Initial Request: " + sql_explanation['is_matching'])

      # If BigQuery validation AND Query semantic validation were both successful : SQL was correctly generated.
      if status['bq_status']['status'] == 'Success' and sql_explanation['is_matching'] == 'True':

        status['sql_generation_success'] = True
        workflow['stop_loop'] = True

      else:  # Failure on either BigQuery SQL validation or query semantic validation
          
        if workflow['stop_loop'] is not True:
          workflow['error_retry_count'] += 1
          if workflow['error_retry_count'] > workflow['error_retry_max_count']:
            workflow['stop_loop'] = True
            status['error_message'] = 'Question cannot be answered using the configured datasets'

        logger.info("Requesting SQL rewrite using chat LLM. Retry number #" + str(workflow['error_retry_count']))
        # append_2_bq_result = bigquery_handler.append_2_bq(
        #   cfg.sql_generation_model_id,
        #   question,
        #   generated_sql, 'N', 'Y',
        #   'explain_plan_validation',
        #   status['bq_status']['error_message'] )

        if workflow['stop_loop'] is not True:

          ### Need to call retry
          if workflow['first_loop'] is True:
            error_correction_chat_session = genai.SQLCorrectionChat()
            workflow['first_loop'] = False

          rewrite_result = execute_with_timeout(
            error_correction_chat_session.get_chat_response,
            table_result_joined,
            similar_questions,
            question,
            generated_sql,
            status['bq_status']['error_message'],
            sql_explanation['mismatch_details'])
          
          streamlit_status.write("New SQL Query Generated. Trial #" + str(workflow['error_retry_count']) + "/" + str(workflow['error_retry_max_count']))
          
          generated_sql = rewrite_result

    # After the loop, check whether is successfully generated a valid SQL query or not
    if status['sql_generation_success'] is False:

      # If the generation was unsuccessful, set the status flags accordingly
      if workflow['error_retry_count'] > workflow['error_retry_max_count']:
        logger.info('Oopss!!! Could not find a valid SQL. This is the best I came up with !!!!!\n' + generated_sql)

      # If query is unrelated to the dataset
      if status['unrelated_question'] is True:
        logger.info('Question cannot be answered using this dataset!')
        status['error'] = 'Error'
        # append_2_bq_result = bigquery_handler.append_2_bq(
        #   cfg.model_id,
        #   question,
        #   'Question cannot be answered using this dataset!',
        #   'N', 'N', 'unrelated_question', '')

      logger.info("Can't find a correct SQL Query that matches exactly the initial questions.")
      status['status'] = 'Error'
      status['error_messages'] = ('BigQuery errors: ' + status['bq_status']['error_message'] if status['bq_status']['status'] != 'Success' else '') \
            + ('Semantic Validation Errors: ' + sql_explanation['mismatch_details'] if sql_explanation['is_matching'] == 'False' else '')

    # If SQL query was successfully generated and tested on BigQuery, proceed with query validation
    else:

      if cfg.execute_final_sql:
        # Query is valid, now executing it against BigQuery
        logger.info('Executing generated and validated SQL on BigQuery')
        status['bq_status'], sql_result_df = bigquery_handler.execute_bq_query(generated_sql, dry_run=False)

        if status['bq_status']['status'] == 'Success':
          logger.info('BigQuery execution successfully completed.')
          status['status'] = 'Success'
          
        else:
          logger.error('BigQuery execution failed. Check the error message for more information. Can\'t answer user query.')
          status['status'] = 'Error'
          status['sql_validation_success'] = False
          status['error_message'] = status['bq_status']['error_message']

      else:
        logger.info('Skipping BigQuery execution as stated in the configuration file.')
        status['status'] = 'Success'

      if cfg.auto_add_knowngood_sql is True:
        #### Adding to the Known Good SQL Vector DB
        logger.info("Adding Known Good SQL to Vector DB...")
        start_time = time.time()
        pgvector_handler.add_vector_sql_collection(cfg.schema, question, generated_sql, question_text_embedding, 'Y')
        metrics['sql_added_to_vector_db_duration'] = time.time() - start_time
        logger.info('SQL added to Vector DB')

  else:
    # Found the record on vector id
    # logger.info('\n Found Question in Vector. Returning the SQL')
    logger.info("Found matching SQL request in pgVector: ", search_sql_vector_by_id_return)
    generated_sql = search_sql_vector_by_id_return
    if cfg.execute_final_sql is True:
        bq_status, final_exec_result_df = bigquery_handler.execute_bq_query(search_sql_vector_by_id_return, dry_run=False)

        #TODO Check BQ response status
        sql_result_df = final_exec_result_df
        logger.info('Question: ' + question)
        logger.info('Final SQL Execution Result:\n' + final_exec_result_df)

    else:  # Do not execute final SQL
        logger.info("Not executing final SQL since EXECUTE_FINAL_SQL variable is False")
    logger.info('will call append to bq next')
    appen_2_bq_result = bigquery_handler.append_2_bq(cfg.model_id, question, search_sql_vector_by_id_return, 'Y', 'N', '', '')

    status['status'] = 'Success'

  if sql_result_df is not None:
    if 'hll_user_aggregates' in matched_tables:
      if sql_result_df.empty is not True:
        sql_result_df.reset_index(drop=True)
      is_audience_result = True
    else:
      is_audience_result = False
  else:
    is_audience_result = False

  response = {
    'status': status['status'],
    'error_message': status['error_message'] if sql_result_df is None else None,
    'generated_sql': '<pre>' + generated_sql + '</pre>' if generated_sql != '' else None,
    'sql_result': sql_result_df,
    'is_audience_result': str(is_audience_result) if sql_result_df is not None else 'False',
    'total_execution_time': round(time.time() - total_start_time, 3),
    'embedding_generation_duration': round(metrics['embedding_duration'], 3),
    'similar_questions_duration': round(metrics['similar_questions_duration'], 3),
    'table_matching_duration': round(metrics['table_matching_duration'], 3),
    'sql_generation_duration': round(metrics['sql_generation_duration'], 3) if metrics['sql_generation_duration'] != -1 else None,
    'bq_execution_duration': round(metrics['bq_execution_duration'], 3) if metrics['bq_execution_duration'] != -1 else None,
    'sql_added_to_vector_db_duration' : round(metrics['sql_added_to_vector_db_duration'], 3) if sql_result_df is not None and cfg.auto_add_knowngood_sql is True else None
  }

  logger.info("Generated object: \n" + str(response))

  return response

# -----------------------------------------------------------------------------
# Module Initialization

# Load the log config file
with open('logging_config.yaml', 'rt') as f:
    config = yaml.safe_load(f.read())

# Configure the logging module with the config file
logging.config.dictConfig(config)

# create logger
global logger
logger = logging.getLogger('nl2sql')
logger.info("Starting nl2sql module")

# Override default uncaught exception handler to log all exceptions using the custom logger
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

logger.info("-------------------------------------------------------------------------------")
logger.info("-------------------------------------------------------------------------------")

pgvector_handler.init()

genai.init()

if cfg.update_db_at_startup is True:
  init_table_and_columns_desc()