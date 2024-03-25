# Common Imports
import pandas_gbq
import pandas as pd
from pathlib import Path
import pgvector_handler
import os, json, time, yaml, sys, logging
import cfg
import genai
import bigquery_handler
from streamlit.elements.lib.mutable_status_container import StatusContainer
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(5)

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
    'error_messages': None,
    'reversed_question' : None
  }

  if streamlit_status is not None:
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
    similar_questions = pgvector_handler.search_sql_nearest_vector(question, question_text_embedding, 'Y')

    metrics['similar_questions_duration'] = time.time() - start_time
    if len(similar_questions) == 0:
      logger.info("No similar questions found...")
    else:
      logger.info("Similar questions found:\n" + str(similar_questions))

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
        if streamlit_status is not None:
          streamlit_status.write("SQL Query Generated")
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
        status['unrelated_question']=True
        status['sql_generation_success'] = False
        logger.info('No ANN/appropriate tables found in Vector to answer the question. Stopping...')

    while workflow['stop_loop'] is False:

      # If similar requests were found, Execute SQL test plan and semantic validation in parallel in order to minimize
      # overall query generation time, as there is a high probability that the correct SQL Query will be found the first iteration.
      logger.info("Executing SQL test plan and SQL query semantic validation in parallel...")

      future_test_plan = executor.submit(bigquery_handler.execute_bq_query, generated_sql, dry_run=True)

      # If validation of the request is requested, execute the GenAI prompt. Otherwise, consider that the generated query matches the question
      if cfg.semantic_validation is True:
        future_sql_validation = executor.submit(
          execute_with_timeout,
          genai.sql_explain,
          question,
          generated_sql,
          table_result_joined,
          similar_questions)
      else:
        sql_explanation = {'is_matching': True, 'reversed_question': None}

      status['bq_status'],_ = future_test_plan.result()
      if streamlit_status is not None:
        streamlit_status.write("SQL Query Syntax Check: " + status['bq_status']['status'])

      if cfg.semantic_validation is True:
        sql_explanation = future_sql_validation.result()
        if streamlit_status is not None:
          streamlit_status.write("SQL Query Matches Initial Request: " + str(sql_explanation['is_matching']))
        status['reversed_question'] = sql_explanation['reversed_question']

      # If BigQuery validation AND Query semantic validation were both successful : SQL was correctly generated.
      if status['bq_status']['status'] == 'Success' and sql_explanation['is_matching'] == True:

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
            error_correction_chat_session = genai.CorrectionSession(table_result_joined, question, similar_questions)
            workflow['first_loop'] = False

          rewrite_result = execute_with_timeout(
            error_correction_chat_session.get_corrected_sql,
            generated_sql,
            status['bq_status']['error_message'],
            sql_explanation['mismatch_details'])
          
          if streamlit_status is not None:
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
        status['error_message'] = 'Question cannot be answered using this dataset!'
        # append_2_bq_result = bigquery_handler.append_2_bq(
        #   cfg.model_id,
        #   question,
        #   'Question cannot be answered using this dataset!',
        #   'N', 'N', 'unrelated_question', '')
      else:
        logger.info("Can't find a correct SQL Query that matches exactly the initial questions.")
        status['status'] = 'Error'
        status['error_message'] = ('BigQuery errors: ' + status['bq_status']['error_message'] if status['bq_status']['status'] != 'Success' else '') \
              + ('Semantic Validation Errors: ' + sql_explanation['mismatch_details'] if sql_explanation['is_matching'] == False else '')

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
        pgvector_handler.add_vector_sql_collection(question, generated_sql, question_text_embedding, 'Y')
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
    # Convert dates columns into datetime dtype, so that they can be easily identified in the UI part
    sql_result_df = sql_result_df.apply(lambda col: pd.to_datetime(col, errors='ignore', format="%Y-%m-%d") if col.dtypes == object else col, axis=0)
    sql_result_df = sql_result_df.apply(lambda col: pd.to_numeric(col, errors='ignore') if col.dtypes == object else col, axis=0)
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
    'reversed_question': status['reversed_question'] if status['reversed_question'] is not None else None,
    'sql_added_to_vector_db_duration' : round(metrics['sql_added_to_vector_db_duration'], 3) if sql_result_df is not None and cfg.auto_add_knowngood_sql is True else None
  }

  logger.info("Generated object: \n" + str(response))

  return response

# -----------------------------------------------------------------------------
# Module Initialization

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

genai.init()