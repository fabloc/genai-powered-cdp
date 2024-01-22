# Common Imports
from datetime import datetime
import pandas_gbq
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from logging import exception
import pgvector_handler
import os, logging, json, time, re
import cfg
import chat_session


class UserSession:
  
  user_history = []

  def __init__(self, prompt_init):
    context_prompt = {
      'role': 'user',
      'parts': [prompt_init]
    }
    self.user_history.append(context_prompt)

  def add_user_question(self, question):
    message = {
      'role': 'user',
      'parts': [question]
    }
    self.user_history.append(message)
  
  def add_model_answer(self, answer):
    message = {
      'role': 'model',
      'parts': [answer]
    }
    self.user_history.append(message)
    
  def get_messages(self):
    return self.history
  
  def reset_history(self):
    self.history = []

class BigQueryError(Exception):
  # Constructor or Initializer
  def __init__(self, type, code, message):
      self.type = type
      self.code = code
      self.message = message

  # __str__ is to print() the value
  def __str__(self):
      return("BigQuery Error with type: " + self.type + ", code: " + self.code + ", and message: " + self.message)

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

def init():

  global logger
  logger = logging.getLogger('nl2sql')

  logger.info("Starting nl2sql module")

  #vertexai.init(project=PROJECT_ID, location="us-central1")
  global model
  model = createModel(cfg.project_id, "us-central1", cfg.model_id)

  global chat_model
  chat_model = createModel(cfg.project_id, cfg.region, cfg.chat_model_id)

  # Enable NL2SQL Analytics Warehouse
  if cfg.enable_analytics is True:
      # Create a BigQuery client
      bq_client = bigquery.Client(location=cfg.dataset_location, project=cfg.project_id)

      # Create a dataset
      try:
        dataset = bq_client.create_dataset(dataset=cfg.dataset_name)
      except Exception as e:
        logger.error('Failed to create the dataset\n')
        logger.error(str(e))

  if cfg.update_db_at_startup is True:
    init_table_and_columns_desc()

# Initialize Palm Models to be used
def createModel(PROJECT_ID, REGION, model_id):
  from vertexai.preview.generative_models import GenerativeModel
  from vertexai.preview.language_models import CodeGenerationModel, ChatModel, CodeChatModel

  if model_id == 'code-bison-32k':
    model = CodeGenerationModel.from_pretrained(model_id)
  elif model_id == 'gemini-pro':
    model = GenerativeModel(model_id)
  elif model_id == 'codechat-bison-32k':
    model = CodeChatModel.from_pretrained(model_id)
  elif model_id == 'chat-bison-32k':
    model = ChatModel.from_pretrained(model_id)
  else:
    raise ValueError
  return model

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
        context_query = generate_sql(context_prompt)
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
    samples_filename = 'queries_samples/' + table_name + '.json'
    if os.path.exists(samples_filename):
      queries_str = open(samples_filename)
      table_queries_samples = json.load(queries_str)
      queries_samples.append(table_queries_samples)
      for sql_query in queries_samples[0]:
        question = sql_query['question']
        question_text_embedding = pgvector_handler.text_embedding(question)
        pgvector_handler.add_vector_sql_collection(cfg.schema, sql_query['question'], sql_query['sql_query'], question_text_embedding, 'Y')
  return queries_samples


def serialized_detailed_description(df):
    detailed_desc = ''
    for index, row in df.iterrows():
        detailed_desc = detailed_desc + str(row['detailed_description']) + '\n'
    return detailed_desc


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

      # Dump the table description
      table_desc = serialized_detailed_description(table_comments_df)

      pgvector_handler.add_table_desc_2_pgvector(table_comments_df)

    # Look for files listing sample queries to be ingested in the pgVector DB
    #insert_sample_queries_lookup(tables_list)


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
      table_cols.append(row['column_name'] + ' (' + row['data_type'] + ') - ' + row['column_description'])

    for index, row in pkeys_df.loc[ (pkeys_df['owner'] == cur_table_owner) & (pkeys_df['table_name'] == cur_table_name)  ].iterrows():
      # Inside each owner.table_name combination
      table_pk_cols.append( row['column_name']  )

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
    [Table]: `{cur_full_table}`
    [Table Description]: {str(row_aug['comments'])}
    [Column (type) - Description]:
    {ln.join(table_cols)}
    [Primary Key]: {final_pk_cols}
    [Foreign Keys]: {final_fk_cols}
    [Owner]: {cur_table_owner}
    [Project_id]: {str(row_aug['project_id'])}
    """

    # Works well
    aug_table_comments_df.at[index_aug, 'detailed_description'] = aug_table_desc
    logger.info("Table schema: \n" + aug_table_desc)
  return aug_table_comments_df


# Build a custom "detailed_description" in the columns dataframe. This will be indexed by the Vector DB
# Augment columns dataframe with detailed description. This detailed description column will be the one used as the document when adding the record to the VectorDB

def build_column_desc(columns_df):
  aug_columns_df = columns_df

  #self.logger.info(len(aug_columns_df))
  #self.logger.info(len(columns_df))

  cur_table_name = ""
  cur_table_owner = ""
  cur_full_table= cur_table_owner + '.' + cur_table_name

  for index_aug, row_aug in aug_columns_df.iterrows():

    cur_table_name = str(row_aug['table_name'])
    cur_table_owner = str(row_aug['owner'])
    cur_full_table= cur_table_owner + '.' + cur_table_name
    curr_col_name = str(row_aug['column_name'])

    #self.logger.info('\n' + cur_table_owner + '.' + cur_table_name + ':')

    col_comments_text=f"""
        Column Name: {row_aug['column_name']} |
        Sample values: {row_aug['sample_values']} |
        Data type: {row_aug['data_type']} |
        Table Name: {row_aug['table_name']} |
        Table Owner: {row_aug['owner']} |
        Project_id: {row_aug['project_id']}
    """
        #Low value: {row_aug['low_value']} |
        #High value: {row_aug['high_value']} |
        #Description: {row_aug['column_description']} |
        #User commments: {row_aug['column_comments']}

    logger.info(' Column ' + cur_full_table + '.' + curr_col_name + " Description: " + col_comments_text)

    aug_columns_df.at[index_aug, 'detailed_description'] = col_comments_text
  return aug_columns_df


def generate_sql(context_prompt):
  generated_sql_json = model.generate_content(
    context_prompt,
    generation_config={
      "max_output_tokens": 2048,
      "temperature": 0,
      "top_p": 1
  })
  generated_sql = generated_sql_json.candidates[0].content.parts[0].text
  return generated_sql


def gen_dyn_rag_sql(question,table_result_joined, similar_questions):
  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'
  context_prompt = f"""

    You are a BigQuery SQL guru. Write a SQL conformant query for Bigquery that answers the following question while using the provided context to correctly refer to the BigQuery tables and the needed column names.

    Guidelines:
    {cfg.prompt_guidelines}

    Tables Schema:
    {table_result_joined}

    {similar_questions}

    Question:
    {question}

    SQL Generated:

    """

    #Column Descriptions:
    #{column_result_joined}

  logger.debug('LLM GEN SQL Prompt: \n' + context_prompt)

  context_query = generate_sql(context_prompt)

  return context_query

def sql_explain(question, generated_sql, table_schema):
  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'
  response_json = {
    'sql_explanation': '{ Write the generated explanation, but don\'t deep-dive into the details of the SQL query }',
    'is_matching': '{Answer with "True" or "False" depending on the outcome of the comparison between generated explanation and the Target Question}',
    'mismatch_details': '{Write all identified mismatch between generated explanation and the Target Question here. If not, return an empty string}'
  }

  context_prompt = f"""You are a BigQuery SQL guru. Generate a high-level semantic explanation of a SQL Query and compare it to a Target Question.
Return the answer as a json object and highlight all the differences identified between the generated explanation and the Target Question.

Guidelines:
    - Analyze the database and the table schema provided as parameters and understand the relations (column and table relations) and the columns description.
    - In the generated explanation, don't deep-dive into the details of the SQL query.
    - When comparing the generated explanation and the Target Question, be as thorough as possible in spotting difference between the generated explanation and the Target Question.
    - If one or more conditions is present in the generated explanation and not in the Target Question, then the generated explanation does not match the Target Question.
    - Remove ```json and ``` from the outputs
    - Answer using the following json format:
    {json.dumps(response_json)}

Tables Schema:
{table_schema}

SQL Query:
{generated_sql}

Target Question:
{question}
"""

  logger.debug('LLM GEN SQL Prompt: \n' + context_prompt)

  context_query = json.loads(generate_sql(context_prompt))

  if context_query['is_matching'] == 'true':
    context_query['is_matching'] = True

  return context_query


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
    logger.info('BQ Analytics is disabled so query was not added to BQ log table')

    return 'BQ Analytics is disabled'



def call_gen_sql(question):

  total_start_time = time.time()
  generated_sql = ''
  sql_result_df = None
  matched_tables = []
  status = 'error'

  embedding_duration = ''
  similar_questions_duration = ''
  table_matching_duration = ''
  sql_generation_duration = ''
  bq_validation_duration = ''
  bq_execution_duration = ''

  logger.info("Creating text embedding from question...")
  start_time = time.time()
  question_text_embedding = pgvector_handler.text_embedding(question)
  embedding_duration = time.time() - start_time
  logger.info("Text embedding created")

  logger.info("User questions: " + str(question))

  # Will look into the Vector DB first and see if there is a hash match.
  # If yes, return the known good SQL.
  # If not, return 3 good examples to be used by the LLM
  #search_sql_vector_by_id_return = pgvector_handler.search_sql_vector_by_id(schema, question,'Y')
  logger.info("Look for exact same question in pgVector...")
  search_sql_vector_by_id_return = 'SQL Not Found in Vector DB'

  # search_sql_vector_by_id_return = "SQL Not Found in Vector DB"

  try:

    if search_sql_vector_by_id_return == 'SQL Not Found in Vector DB':   ### Only go thru the loop if hash of the question is not found in Vector.

      logger.info("Did not find same question in DB")
      logger.info("Searching for similar questions in DB...")
      start_time = time.time()
      # Look into Vector for similar queries. Similar queries will be added to the LLM prompt (few shot examples)
      similar_questions_return = pgvector_handler.search_sql_nearest_vector(cfg.schema, question, question_text_embedding, 'Y')
      similar_questions_duration = time.time() - start_time
      logger.info("Found similar questions:\n" + str(similar_questions_return))

      unrelated_question=False
      stop_loop = False
      error_retry_max_count= cfg.sql_max_error_retry
      explanation_retry_max_count = cfg.sql_max_explanation_retry
      error_retry_count=0
      explanation_retry_count = 0
      error_correction_chat_session=None
      explanation_correction_chat_session=None
      logger.info("Now looking for appropriate tables in Vector to answer the question...")
      start_time = time.time()
      table_result_joined = pgvector_handler.get_tables_colums_vector(question, question_text_embedding)
      table_matching_duration = time.time() - start_time

      if len(table_result_joined) > 0 :
          logger.info("Found matching tables")

          # Add matched table to list of tables used during the SQL generation
          for table in cfg.tables:
            if table in table_result_joined:
              matched_tables.append(table)

          start_time = time.time()
          logger.info("Generating SQL query using LLM...")
          generated_sql=gen_dyn_rag_sql(question,table_result_joined, similar_questions_return)
          logger.info("SQL query generated:\n" + generated_sql)
          sql_generation_duration = time.time() - start_time
          if 'unrelated_answer' in generated_sql :
            stop_loop=True
            #logger.info('Inside if statement to check for unrelated question...')
            unrelated_question=True
          if cfg.inject_one_error is True:
            if error_retry_count < 1:
              logger.info('Injecting error on purpose to test code ... Adding ROWID at the end of the string')
              generated_sql=generated_sql + ' ROWID'
      else:
          stop_loop=True
          unrelated_question=True
          logger.info('No ANN/appropriate tables found in Vector to answer the question. Stopping...')

      while (stop_loop is False):

        start_time = time.time()
        logger.info('Executing SQL query...')
        bq_query_execution_status = 'Success'
        start_time = time.time()

        try:
          sql_result_df=execute_sql(generated_sql)
          bq_query_execution_status = 'Success'
        except BigQueryError as error:
          if error.type == 'Bad SQL query':
            logger.error("BigQuery query execution failed with error code: " + str(error.code) + " and error message: " + error.message)
            bq_query_execution_status = error.message
          elif error.type == 'Permission Error':
            logger.error("Permission error while trying to access BigQuery Table with error code: " + str(error.code) + " and error message: " + error.message)
            raise Exception(err)
          else:
            raise Exception(err)
        except Exception as error:
          print(error)
        finally:
          bq_execution_duration = start_time + time.time()
          logger.info("SQL execution complete")

        if bq_query_execution_status == 'Success':
          stop_loop = True

          logger.info('Question: ' + question)
          logger.info('SQL Execution Result:\n' + str(sql_result_df))


          if len(sql_result_df) >= 1:
            if not "ORA-" in str(sql_result_df.iloc[0,0]):
                # Check whether the generated query actually answers the initial question by invoking GenAI
                # to analyze what the generated SQL is doing. It returns a json containing the result of the analysis
                sql_explanation = sql_explain(question, generated_sql, table_result_joined)
                logger.info('Generated SQL explanation: ' + json.dumps(sql_explanation))
                if 'is_matching' in sql_explanation and sql_explanation['is_matching'] == 'True':
                  logger.info("Generated SQL explanation matches initial question.")
                  if cfg.auto_add_knowngood_sql is True:  #### Adding to the Known Good SQL Vector DB
                    logger.info("Adding Known Good SQL to Vector DB...")
                    start_time = time.time()
                    pgvector_handler.add_vector_sql_collection(cfg.schema, question, generated_sql, question_text_embedding, 'Y')
                    sql_added_to_vector_db_duration = time.time() - start_time
                    logger.info('SQL added to Vector DB')
                  status = 'success'
                else:
                  # If generated SQL does not match initial question, start again by asking GenAI model to adapt the query
                  stop_loop = False
                  if explanation_correction_chat_session is None: explanation_correction_chat_session = chat_session.ExplanationCorrectionChat(model)
                  generated_sql = explanation_correction_chat_session.get_chat_response(table_result_joined, similar_questions_return, question, generated_sql, sql_explanation['sql_explanation'], sql_explanation['mismatch_details'])
                  explanation_retry_count+=1
                  error_retry_count = 0
            else:
                ### Need to call retry
                stop_loop = False
                if error_correction_chat_session is None: error_correction_chat_session = chat_session.SQLCorrectionChat(model)
                generated_sql = error_correction_chat_session.get_chat_response(question, generated_sql, table_result_joined, bq_query_execution_status,similar_questions_return)
                error_retry_count+=1

          appen_2_bq_result = append_2_bq(cfg.model_id, question, generated_sql, 'N', 'N', '', '')

        else:  # Failure on BigQuery SQL execution
            logger.info("Error during SQL execution")
            logger.info("Requesting SQL rewrite using chat LLM. Retry number #" + str(error_retry_count))
            append_2_bq_result = append_2_bq(cfg.model_id, question, generated_sql, 'N', 'Y', 'explain_plan_validation', bq_query_execution_status )
            ### Need to call retry
            if error_correction_chat_session is None: error_correction_chat_session = chat_session.SQLCorrectionChat(model)
            rewrite_result = error_correction_chat_session.get_chat_response(question, generated_sql, table_result_joined, bq_query_execution_status, similar_questions_return)
            logger.info('\n Rewritten SQL:\n' + rewrite_result)
            generated_sql=rewrite_result
            error_retry_count+=1

        if stop_loop != True:
          if error_retry_count > error_retry_max_count:
            stop_loop = True
            error_message = "Can't correct generated SQL query."
          
          if explanation_retry_count > explanation_retry_max_count:
            stop_loop = True
            error_message = "Can't correct irrelevant SQL query."

      # After the while is completed
      if error_retry_count > error_retry_max_count:
        logger.info('Oopss!!! Could not find a valid SQL. This is the best I came up with !!!!!\n' + generated_sql)

      if explanation_retry_count > explanation_retry_max_count:
        logger.info('Oopss!!! Could not find a SQL exactly matching the question. This is the best I came up with !!!!!\n' + generated_sql)

      # If query is unrelated to the dataset
      if unrelated_question is True:
        logger.info('Question cannot be answered using this dataset!')
        error_message = 'Question cannot be answered using the configured datasets'
        sql_generation_duration = -1
        bq_execution_duration = -1
        append_2_bq_result = append_2_bq(
          cfg.model_id,
          question,
          'Question cannot be answered using this dataset!',
          'N', 'N', 'unrelated_question', '')

    else:
      # Found the record on vector id
      # logger.info('\n Found Question in Vector. Returning the SQL')
      logger.info("Found matching SQL request in pgVector: ", search_sql_vector_by_id_return)
      generated_sql = search_sql_vector_by_id_return
      if cfg.execute_final_sql is True:
          final_exec_result_df = execute_sql(search_sql_vector_by_id_return)
          sql_result_df = final_exec_result_df
          logger.info('Question: ' + question)
          logger.info('Final SQL Execution Result:\n' + final_exec_result_df)

      else:  # Do not execute final SQL
          logger.info("Not executing final SQL since EXECUTE_FINAL_SQL variable is False")
      logger.info('will call append to bq next')
      appen_2_bq_result = append_2_bq(cfg.model_id, question, search_sql_vector_by_id_return, 'Y', 'N', '', '')

      status = 'success'

    if sql_result_df is not None:
      if 'hll_user_aggregates' in matched_tables:
        sql_result_str = "Audience Size: " + str(sql_result_df.iat[0,0].item())
        is_audience_result = True
      else:
        sql_result_str = sql_result_df.to_html()
        is_audience_result = True
    else:
      sql_result_str = ''
      is_audience_result = False

    response = {
      'status': status,
      'error_message': error_message if sql_result_df is None else None,
      'generated_sql': '<pre>' + generated_sql + '</pre>' if generated_sql != '' else None,
      'sql_result': sql_result_str if sql_result_df is not None else None,
      'is_audience_result': str(is_audience_result) if sql_result_df is not None else 'False',
      'total_execution_time': round(time.time() - total_start_time, 3),
      'embedding_generation_duration': round(embedding_duration, 3),
      'similar_questions_duration': round(similar_questions_duration, 3),
      'table_matching_duration': round(table_matching_duration, 3),
      'sql_generation_duration': round(sql_generation_duration, 3) if sql_generation_duration != -1 else None,
      'bq_execution_duration': round(bq_execution_duration, 3) if bq_execution_duration != -1 else None,
      'sql_added_to_vector_db_duration' : round(sql_added_to_vector_db_duration, 3) if sql_result_df is not None and cfg.auto_add_knowngood_sql is True else None
    }

    logger.info("Generated object: \n" + str(response))

    return response
  
  except BigQueryError as err:
    response = {
      'status': 'error',
      'type': err.type,
      'error_message': err.message
    }

    return response
  except Exception as err:
    print("Exception raised: " + str(err.args))


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

    except Exception as err:
      
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
        raise BigQueryError(type='Permission Error', code=error_code, message=error_message)
      elif error_code == 400 or error_code == 404:
        raise BigQueryError(type='Bad SQL query', code=error_code, message=error_message)
      else:
        raise BigQueryError(type='Undefined BigQuery Error', code=error_code, message=error_message_raw)

  # If the maximum number of retries has been reached, raise an exception.
  raise BigQueryError(type='BigQuery Internal Error', code='500',message='Maximum number of retries reached.')