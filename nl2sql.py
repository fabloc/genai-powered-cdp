# Common Imports
import time
from datetime import datetime
import vertexai
import pandas_gbq
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd
import json
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from logging import exception
import asyncio
import pgvector_handler
import os
import logging
import logging.config
import yaml
import sys
import sqlparse

# Load the config file
with open('logging_config.yaml', 'rt') as f:
    config = yaml.safe_load(f.read())

# Configure the logging module with the config file
logging.config.dictConfig(config)

# create logger
logger = logging.getLogger('nl2sql')

# Override default uncaught exception handler to log all exceptions using the custom logger
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

source_type='BigQuery'

# @markdown Provide the below details to start using the notebook
PROJECT_ID='cdp-demo-flocquet'
REGION = 'europe-west1'
DATAPROJECT_ID='cdp-demo-flocquet'
AUTH_USER='admin@fabienlocquet.altostrat.com'
# TABLES contain the tables to be used for SQL generation. Enter an empty table list when all tables in the
# dataset should be used
TABLES=['hll_user_aggregates', 'products_aggregates']

# BQ Schema (DATASET) where tables leave
schema='publisher_1_dataset' ### DDL extraction performed at this level, for the entire schema
USER_DATASET= DATAPROJECT_ID + '.' + schema

# Execution Parameters
SQL_VALIDATION='ALL'
INJECT_ONE_ERROR=False
EXECUTE_FINAL_SQL=True
SQL_MAX_FIX_RETRY=3
AUTO_ADD_KNOWNGOOD_SQL=True

# Analytics Warehouse
ENABLE_ANALYTICS=False
DATASET_NAME='nl2sql'
DATASET_LOCATION='EU'
LOG_TABLE_NAME='query_logs'


# Palm Models to use
model_id='gemini-pro'
chat_model_id='codechat-bison-32k'

# Initialize Palm Models to be used
def createModel(PROJECT_ID, REGION, model_id):
  from vertexai.preview.generative_models import GenerativeModel
  from vertexai.preview.language_models import CodeGenerationModel, ChatModel, CodeChatModel

  if model_id == 'code-bison-32k':
    model = CodeGenerationModel.from_pretrained('code-bison-32k')
  elif model_id == 'gemini-pro':
    model = GenerativeModel(model_id)
  elif model_id == 'codechat-bison-32k':
    model = CodeChatModel.from_pretrained("codechat-bison-32k")
  elif model_id == 'chat-bison-32k':
    model = ChatModel.from_pretrained("chat-bison-32k")
  else:
    raise ValueError
  return model

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

# Define BigQuery Dictionary Queries and Helper Functions

get_columns_sql='''
 SELECT
    columns.TABLE_CATALOG as project_id, columns.TABLE_SCHEMA as owner , columns.TABLE_NAME as table_name, columns_field_paths.FIELD_PATH as column_name,
    columns.IS_NULLABLE as is_nullable, columns_field_paths.DATA_TYPE as data_type, columns.COLUMN_DEFAULT as column_default, columns.ROUNDING_MODE as rounding_mode, DESCRIPTION as column_description
  FROM
    {USER_DATASET}.INFORMATION_SCHEMA.COLUMNS AS columns
  JOIN
    {USER_DATASET}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS AS columns_field_paths
  ON columns.TABLE_CATALOG = columns_field_paths.TABLE_CATALOG AND columns.TABLE_SCHEMA = columns_field_paths.TABLE_SCHEMA
    AND columns.TABLE_NAME = columns_field_paths.TABLE_NAME AND columns.COLUMN_NAME = columns_field_paths.COLUMN_NAME
  WHERE
    CASE
        WHEN ARRAY_LENGTH({TABLES}) > 0 THEN columns.table_name IN UNNEST({TABLES})
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
{USER_DATASET}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS T
JOIN {USER_DATASET}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE K
ON K.CONSTRAINT_NAME=T.CONSTRAINT_NAME
WHERE
T.CONSTRAINT_TYPE="FOREIGN KEY" AND 
(CASE
    WHEN ARRAY_LENGTH({TABLES}) > 0 THEN T.table_name IN UNNEST({TABLES})
    ELSE TRUE END)
ORDER BY
project_id, owner, table_name
'''

get_pkeys_sql='''
SELECT T.CONSTRAINT_CATALOG, T.CONSTRAINT_SCHEMA, T.CONSTRAINT_NAME,
T.TABLE_CATALOG as project_id, T.TABLE_SCHEMA as owner, T.TABLE_NAME as table_name, T.CONSTRAINT_TYPE,
T.IS_DEFERRABLE, T.ENFORCED, K.COLUMN_NAME
FROM
    {USER_DATASET}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS T
JOIN {USER_DATASET}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE K
ON K.CONSTRAINT_NAME=T.CONSTRAINT_NAME
WHERE
    T.CONSTRAINT_TYPE="PRIMARY KEY" AND
    (CASE
        WHEN ARRAY_LENGTH({TABLES}) > 0 THEN T.table_name IN UNNEST({TABLES})
        ELSE TRUE END)
ORDER BY
project_id, owner, table_name
'''


get_table_comments_sql='''
select TABLE_CATALOG as project_id, TABLE_SCHEMA as owner, TABLE_NAME as table_name, OPTION_NAME, OPTION_TYPE, OPTION_VALUE as comments
FROM
    {USER_DATASET}.INFORMATION_SCHEMA.TABLE_OPTIONS
WHERE
    OPTION_NAME = "description" AND
    (CASE
        WHEN ARRAY_LENGTH({TABLES}) > 0 THEN table_name IN UNNEST({TABLES})
        ELSE TRUE END)
ORDER BY
project_id, owner, table_name
'''



def schema_generator(sql):
  formatted_sql = sql.format(**globals(), **locals())
  logger.info("BigQuery request: " + formatted_sql)
  df = pandas_gbq.read_gbq(formatted_sql, project_id=PROJECT_ID, location='europe-west1')
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


def add_column_comments(columns_df, pkeys_df, fkeys_df, table_comments_df):
  for index, row in columns_df.iterrows():
    if row['column_description'] is None: ## or row['comments'] is not None:
        context_prompt = f"""
        Generate comments for the column {row['project_id']}.{row['owner']}.{row['table_name']}.{row['column_name']}


        Parameters:
        - column metadata: {columns_df.to_markdown(index = False)}
        - primary key metadata: {pkeys_df.to_markdown(index = False)}
        - foreign keys metadata: {fkeys_df.to_markdown(index = False)}
        - table metadata: {table_comments_df.to_markdown(index = False)}

      """
        context_query = generate_sql(context_prompt)
        columns_df.at[index, 'column_comments'] = context_query
        #columns_df.at[index, 'column_comments'] = clean_sql("my comments")

  return columns_df

def get_column_sample(columns_df):
  sample_column_list=[]

  for index, row in columns_df.iterrows():
    get_column_sample_sql=f'''
        SELECT STRING_AGG(CAST(value AS STRING)) as sample_values
        FROM UNNEST((SELECT APPROX_TOP_COUNT(TO_JSON_STRING({row["column_name"]}), 5) as osn
                    FROM `{row["project_id"]}.{row["owner"]}.{row["table_name"]}`
                ))
    '''
    try:
      column_samples_df=schema_generator(get_column_sample_sql)
      sample_column_list.append(column_samples_df['sample_values'].to_string(index=False))
    except Exception as err:
      column_name = str(row["column_name"])
      if "Reason: 400 Cannot access field" in err.args[0] and ("STRUCT" in err.args[0] or "ARRAY" in err.arg[0]):
        logger.info("Cannot parse column '" + column_name + "' with type STRUCT and/or ARRAY, skipping")
      else:
        logger.error("Error gathering samples for column " + str(row["column_name"]) + " - with error: " + str(err[0]))
      sample_column_list.append('')


  columns_df["sample_values"]=sample_column_list
  return columns_df

def serialized_detailed_description(df):
    detailed_desc = ''
    for index, row in df.iterrows():
        detailed_desc = detailed_desc + str(row['detailed_description']) + '\n'
    return detailed_desc

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
        pgvector_handler.add_vector_sql_collection(schema, sql_query['question'], sql_query['sql_query'], question_text_embedding, 'Y')
  return queries_samples


def init_table_and_columns_desc():
  
    table_comments_df=schema_generator(get_table_comments_sql)

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
      TABLES=uninitialized_tables
      columns_df=schema_generator(get_columns_sql)
      fkeys_df=schema_generator(get_fkeys_sql)
      pkeys_df=schema_generator(get_pkeys_sql)

      # Adding column sample_values to the columns dataframe
      columns_df=get_column_sample(columns_df)

      # Look at each tables dataframe row and use LLM to generate a table comment, but only for the tables with null comments (DB did not have comments on table)
      ## Using Palm to add table comments if comments are null
      table_comments_df=add_table_comments(columns_df, pkeys_df, fkeys_df, table_comments_df)

      table_comments_df=build_table_desc(table_comments_df,columns_df,pkeys_df,fkeys_df)

      # Dump the table description
      table_desc = serialized_detailed_description(table_comments_df)

      file = open("table_desc.txt", "w")
      file.write(table_desc)
      file.close()

      pgvector_handler.add_table_desc_2_pgvector(table_comments_df)

      # Look at each column in the columns dataframe use LLM to generate a column comment, if one does not exist
      columns_df=add_column_comments(columns_df, pkeys_df, fkeys_df, table_comments_df)

      columns_df=build_column_desc(columns_df)

      column_desc = serialized_detailed_description(columns_df)

      file = open("column_desc.txt", "w")
      file.write(column_desc)
      file.close()

      pgvector_handler.add_column_desc_2_pgvector(columns_df)

    # Look for files listing sample queries to be ingested in the pgVector DB
    insert_sample_queries_lookup(tables_list)


# Build a custom "detailed_description" table column to be indexed by the Vector DB
# Augment Table dataframe with detailed description. This detailed description column will be the one used as the document when adding the record to the VectorDB
def build_table_desc(table_comments_df,columns_df,pkeys_df,fkeys_df):
  aug_table_comments_df = table_comments_df

  #logger.info(len(aug_table_comments_df))
  #logger.info(len(table_comments_df))

  cur_table_name = ""
  cur_table_owner = ""
  cur_project_id = ""
  cur_full_table= cur_project_id + '.' + cur_table_owner + '.' + cur_table_name

  for index_aug, row_aug in aug_table_comments_df.iterrows():

    cur_table_name = str(row_aug['table_name'])
    cur_table_owner = str(row_aug['owner'])
    cur_project_id = str(row_aug['project_id'])
    cur_full_table= cur_project_id + '.' + cur_table_owner + '.' + cur_table_name
    #logger.info('\n' + cur_table_owner + '.' + cur_table_name + ':')

    table_cols=[]
    table_cols_datatype=[]
    table_cols_description=[]
    table_pk_cols=[]
    table_fk_cols=[]

    for index, row in columns_df.loc[ (columns_df['owner'] == cur_table_owner) & (columns_df['table_name'] == cur_table_name) ].iterrows():
      # Inside each owner.table_name combination
      table_cols.append( row['column_name']  )
      table_cols_datatype.append( row['column_name'] + ' (' + row['data_type'] + ') '  )
      table_cols_description.append( '\n        - ' + row['column_name'] + ' (' + row['column_description'] + ')'  )

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

    aug_table_desc=f"""
      Table Name: {cur_full_table} |
      Owner: {cur_table_owner} |
      Schema Columns:{", ".join(table_cols)} |
      Column Types: {", ".join(table_cols_datatype)} |
      Column Descriptions: {"".join(table_cols_description)} |
      Primary Key: {final_pk_cols} |
      Foreign Keys: {final_fk_cols} |
      Project_id: {str(row_aug['project_id'])} |
      Table Comments: {str(row_aug['comments'])}
    """

    #logger.info ('Current aug dataset row: '  + str(row_aug['table_name']))
    #logger.info(aug_table_desc)

    # Works well
    aug_table_comments_df.at[index_aug, 'detailed_description'] = aug_table_desc
  return aug_table_comments_df

# Build a custom "detailed_description" in the columns dataframe. This will be indexed by the Vector DB
# Augment columns dataframe with detailed description. This detailed description column will be the one used as the document when adding the record to the VectorDB

def build_column_desc(columns_df):
  aug_columns_df = columns_df

  #logger.info(len(aug_columns_df))
  #logger.info(len(columns_df))

  cur_table_name = ""
  cur_table_owner = ""
  cur_full_table= cur_table_owner + '.' + cur_table_name

  for index_aug, row_aug in aug_columns_df.iterrows():

    cur_table_name = str(row_aug['table_name'])
    cur_table_owner = str(row_aug['owner'])
    cur_full_table= cur_table_owner + '.' + cur_table_name
    curr_col_name = str(row_aug['column_name'])

    #logger.info('\n' + cur_table_owner + '.' + cur_table_name + ':')

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


# Enable NL2SQL Analytics Warehouse
if ENABLE_ANALYTICS is True:
  # Create a BigQuery client
  bq_client = bigquery.Client(location=DATASET_LOCATION, project=PROJECT_ID)

  # Create a dataset
  try:
    dataset = bq_client.create_dataset(dataset=DATASET_NAME)
  except Exception as e:
    logger.error('Failed to create the dataset\n')
    logger.error(str(e))


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

def chat_send_message(chat_session, context_prompt):
  generated_text = chat_session.send_message(context_prompt)
  logger.info("Generated text: " + generated_text)
  return generated_text.text


def clean_sql(result):
  result = result.replace("```sql", "").replace("```", "")
  return result

def clean_json(result):
  result = result.replace("```json", "").replace("```", "")
  return result

def gen_dyn_rag_sql(question,table_result_joined, similar_questions):
  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'
  context_prompt = f"""

      You are a BigQuery SQL guru. Write a SQL conformant query for Bigquery that answers the following question while using the provided context to correctly refer to the BigQuery tables and the needed column names.

      Guidelines:
      - Only answer questions relevant to the tables listed in the table schema. If a non-related question comes, answer exactly: {not_related_msg}
      - Join as minimal tables as possible.
      - When joining tables ensure all join columns are the same data_type.
      - Analyze the database and the table schema provided as parameters and undestand the relations (column and table relations).
      - When asked to count the number of users, always perform an estimation using Hyperloglog++ (HLL) sketches using HLL_COUNT.MERGE.
      - For all requests not related to the number of users matching certain criteria, never use estimates like HyperLogLog++ (HLL) sketches
      - Never use GROUP BY on HLL sketches.
      - Never use HLL_COUNT.EXTRACT or HLL_COUNT.MERGE inside a WHERE statement.
      - HLL_COUNT.EXTRACT must be used only for HLL sketches.
      - Consider alternative options to CAST function. If performing a CAST, use only Bigquery supported datatypes.
      - Don't include any comments in code.
      - Remove ```sql and ``` from the output and generate the SQL in single line.
      - Tables should be refered to using a fully qualified name (project_id.owner.table_name).
      - Use all the non-aggregated columns from the "SELECT" statement while framing "GROUP BY" block.
      - Return syntactically and semantically correct SQL for BigQuery with proper relation mapping i.e project_id, owner, table and column relation.
      - Use ONLY the column names (column_name) mentioned in Table Schema. DO NOT USE any other column names outside of this.
      - Associate column_name mentioned in Table Schema only to the table_name specified under Table Schema.
      - Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed.
      - Table names are case sensitive. DO NOT uppercase or lowercase the table names.
      - Column names are case sensitive. DO NOT uppercase or lowercase the column names.
      - Owner (dataset) is case sensitive. DO NOT uppercase or lowercase the owner.
      - Project_id is case sensitive. DO NOT uppercase or lowercase project_id.

    Table Schema:
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


    # Question: Display a count of orders by customer_email
    # Generated SQL:
    # SELECT u.email as email, count(*) as count_oders
    # FROM
    # `bigquery-public-data.thelook_ecommerce.orders` o,`bigquery-public-data.thelook_ecommerce.users` u
    #   where o.user_id = u.id
    #   group by email;


def test_sql_plan_execution(generated_sql):
  from google.cloud import bigquery
  try:

    run_dataset=PROJECT_ID + '.' + DATASET_NAME
    df=pd.DataFrame()

    # Construct a BigQuery client object.
    client = bigquery.Client(project=PROJECT_ID)

    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

    # Start the query, passing in the extra configuration.
    query_job = client.query(
        (generated_sql),
        job_config=job_config,
    )  # Make an API request.

    # A dry run query completes immediately.
    logger.info("This query will process {} bytes.".format(query_job.total_bytes_processed))
    return 'Execution Plan OK'
  except Exception as e:
    logger.error(e)
    msg= str(e.errors[0]['message'])
    return msg


def init_chat():
  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'

  context_prompt = f"""

    You are a BigQuery SQL guru. This session is trying to troubleshoot a Google BigQuery SQL query.
    As the user provides versions of the query and the errors returned by BigQuery,
    return a never seen alternative SQL query that fixes the errors.
    It is important that the query still answer the original question.

      Guidelines:
      - Only answer questions relevant to the tables listed in the table schema. If a non-related question comes, answer exactly: {not_related_msg}
      - Join as minimal tables as possible.
      - When joining tables ensure all join columns are the same data_type.
      - Analyze the database and the table schema provided as parameters and undestand the relations (column and table relations).
      - When asked to count the number of users, always perform an estimation using Hyperloglog++ (HLL) sketches using HLL_COUNT.MERGE.
      - For all requests not related to the number of users matching certain criteria, never use estimates like HyperLogLog++ (HLL) sketches
      - Never use GROUP BY on HLL sketches.
      - Never use HLL_COUNT.EXTRACT or HLL_COUNT.MERGE inside a WHERE statement.
      - HLL_COUNT.EXTRACT must be used only for HLL sketches.
      - Consider alternative options to CAST function. If performing a CAST, use only Bigquery supported datatypes.
      - Don't include any comments in code.
      - Remove ```sql and ``` from the output and generate the SQL in single line.
      - Tables should be refered to using a fully qualified name (project_id.owner.table_name).
      - Use all the non-aggregated columns from the "SELECT" statement while framing "GROUP BY" block.
      - Return syntactically and semantically correct SQL for BigQuery with proper relation mapping i.e project_id, owner, table and column relation.
      - Use ONLY the column names (column_name) mentioned in Table Schema. DO NOT USE any other column names outside of this.
      - Associate column_name mentioned in Table Schema only to the table_name specified under Table Schema.
      - Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed.
      - Table names are case sensitive. DO NOT uppercase or lowercase the table names.
      - Owner (dataset) is case sensitive. DO NOT uppercase or lowercase the owner.
      - Project_id is case sensitive. DO NOT uppercase or lowercase the project_id.

  """
  logger.info('Initializing code chat model ...')
  chat_session = chat_model.start_chat(context=context_prompt)
  logger.info('Code chat model initialized')
  # context_prompt
  return chat_session



def rewrite_sql_chat(chat_session, question, generated_sql, table_result_joined, error_msg, similar_questions):

  context_prompt = f"""
    What is an alternative SQL statement to address the error mentioned below?
    Present a different SQL from previous ones. It is important that the query still answer the original question.
    Do not repeat suggestions.

  Question:
  {question}

  Previously Generated (bad) SQL Query:
  {generated_sql}

  Error Message:
  {error_msg}

  Table Schema:
  {table_result_joined}

  {similar_questions}
  """

  #Column Descriptions:
  #{column_result_joined}

  logger.info("SQL rewriting prompt:\n", context_prompt, "\n'")

  response = chat_send_message(chat_session, context_prompt)

  #logger.info(str(response))
  return clean_sql(response)


#question="Display the result of selecting test word from dual"
#final_sql='select \'test\' from dual'
#ret=add_vector_sql_collection('HR', question, final_sql, 'Y')
#logger.info( ret )


def append_2_bq(model, question, generated_sql, found_in_vector, need_rewrite, failure_step, error_msg):

  if ENABLE_ANALYTICS is True:
      logger.debug('\nInside the Append to BQ block\n')
      table_id=PROJECT_ID + '.' + DATASET_NAME + '.' + LOG_TABLE_NAME
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
          "source_type":source_type,
          "project_id":str(PROJECT_ID),
          "user":str(AUTH_USER),
          "schema": schema,
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
        #logger.info("Table {} already exists.".format(table_id))
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
          pandas_gbq.to_gbq(df1, table_id, project_id=PROJECT_ID)  # replace to replace table; append to append to a table


      # df1.loc[len(df1)] = new_row
      # pandas_gbq.to_gbq(df1, table_id, project_id=PROJECT_ID, if_exists='append')  # replace to replace table; append to append to a table
      logger.info('Query added to BQ log table')
      return 'Row added'
  else:
    logger.info('BQ Analytics is disabled so query was not added to BQ log table')

    return 'BQ Analytics is disabled'



def call_gen_sql(question):

  total_start_time = time.time()
  generated_valid_sql = ''
  sql_result = ''

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

  # Overwriting for testing purposes
  #INJECT_ONE_ERROR = True

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
        similar_questions_return = pgvector_handler.search_sql_nearest_vector(schema, question, question_text_embedding, 'Y')
        similar_questions_duration = time.time() - start_time
        logger.info("Found similar questions:\n" + str(similar_questions_return))

        unrelated_question=False
        stop_loop = False
        retry_max_count= SQL_MAX_FIX_RETRY
        retry_count=0
        chat_session=None
        logger.info("Now looking for appropriate tables in Vector to answer the question...")
        start_time = time.time()
        table_result_joined = pgvector_handler.get_tables_colums_vector(question, question_text_embedding)
        table_matching_duration = time.time() - start_time

        if len(table_result_joined) > 0 :
            logger.info("Found matching tables")
            start_time = time.time()
            logger.info("Generating SQL query using LLM...")
            generated_sql=gen_dyn_rag_sql(question,table_result_joined, similar_questions_return)
            
            sql_generation_duration = time.time() - start_time
            if 'unrelated_answer' in generated_sql :
              stop_loop=True
              #logger.info('Inside if statement to check for unrelated question...')
              unrelated_question=True
        else:
            stop_loop=True
            unrelated_question=True
            logger.info('No ANN/appropriate tables found in Vector to answer the question. Stopping...')



        while (stop_loop is False):

          if INJECT_ONE_ERROR is True:
            if retry_count < 1:
              logger.info('Injecting error on purpose to test code ... Adding ROWID at the end of the string')
              generated_sql=generated_sql + ' ROWID'

          start_time = time.time()
          logger.info("Testing code execution by performing explain plan on SQL...")
          sql_plan_test_result=test_sql_plan_execution(generated_sql) # Calling explain plan
          logger.info('Dry-run complete')
          bq_validation_duration = time.time() - start_time

          logger.debug("BigQuery explain plan result:\n" + sql_plan_test_result)

          if sql_plan_test_result == 'Execution Plan OK':  # Explain plan is OK

            logger.info("BigQuery explain plan successful")

            generated_valid_sql = generated_sql

            stop_loop = True

            if EXECUTE_FINAL_SQL is True:
              start_time = time.time()
              logger.info('Executing SQL query...')
              final_exec_result_df=execute_final_sql(generated_sql)
              logger.info('SQL query complete')
              bq_execution_duration = time.time() - start_time
              logger.info('Question: ' + question)
              logger.info('Final SQL Execution Result:\n' + str(final_exec_result_df))
              sql_result = final_exec_result_df
              if AUTO_ADD_KNOWNGOOD_SQL is True:  #### Adding to the Known Good SQL Vector DB
                if len(final_exec_result_df) >= 1:
                  if not "ORA-" in str(final_exec_result_df.iloc[0,0]):
                      logger.info('Adding Known Good SQL to Vector DB...')
                      start_time = time.time()
                      pgvector_handler.add_vector_sql_collection(schema, question, generated_sql, question_text_embedding, 'Y')
                      sql_added_to_vector_db_duration = time.time() - start_time
                      logger.info('SQL added to Vector DB')
                  else:
                      ### Need to call retry
                      stop_loop = False
                      if chat_session is None: chat_session=init_chat()
                      rewrite_result=rewrite_sql_chat(chat_session, question, generated_sql, table_result_joined, str(final_exec_result_df.iloc[0,0]) ,similar_questions_return)
                      logger.info('Rewritten SQL:\n' + rewrite_result)
                      generated_sql=rewrite_result
                      retry_count+=1


            else:  # Do not execute final SQL
              logger.info("Not executing final SQL since EXECUTE_FINAL_SQL variable is False\n ")

            appen_2_bq_result=append_2_bq(model_id, question, generated_sql, 'N', 'N', '', '')

          else:  # Failure on explain plan execution
              logger.info("Error on explain plan execution")
              logger.info("Requesting SQL rewrite using chat LLM. Retry number #" + str(retry_count))
              append_2_bq_result=append_2_bq(model_id, question, generated_sql, 'N', 'Y', 'explain_plan_validation', sql_plan_test_result )
              ### Need to call retry
              if chat_session is None: chat_session=init_chat()
              rewrite_result=rewrite_sql_chat(chat_session, question, generated_sql, table_result_joined, sql_plan_test_result,similar_questions_return)
              logger.info('\n Rewritten SQL:\n' + rewrite_result)
              generated_sql=rewrite_result
              retry_count+=1

          if retry_count > retry_max_count:
            stop_loop = True

        # After the while is completed
        if retry_count > retry_max_count:
          logger.info('Oopss!!! Could not find a good SQL. This is the best I came up with !!!!!\n' + generated_sql)

        # If query is unrelated to the dataset
        if unrelated_question is True:
          logger.info('Question cannot be answered using this dataset!')
          append_2_bq_result=append_2_bq(model_id, question, 'Question cannot be answered using this dataset!', 'N', 'N', 'unrelated_question', '')

          #logger.info(generated_sql)

  else:   ## Found the record on vector id
    #logger.info('\n Found Question in Vector. Returning the SQL')
    logger.info("Found matching SQL request in pgVector: ", search_sql_vector_by_id_return)
    generated_valid_sql = search_sql_vector_by_id_return
    if EXECUTE_FINAL_SQL is True:
        final_exec_result_df=execute_final_sql(search_sql_vector_by_id_return)
        sql_result = final_exec_result_df
        logger.info('Question: ' + question)
        logger.info('Final SQL Execution Result:\n' + final_exec_result_df)

    else:  # Do not execute final SQL
        logger.info("Not executing final SQL since EXECUTE_FINAL_SQL variable is False")
    logger.info('will call append to bq next')
    appen_2_bq_result=append_2_bq(model_id, question, search_sql_vector_by_id_return, 'Y', 'N', '', '')

  response = {
    'generated_sql': sqlparse.format(generated_valid_sql, reindent=True, keyword_case='upper'),
    'sql_result': str(sql_result),
    'total_execution_time': round(time.time() - total_start_time, 3),
    'embedding_generation_duration': round(embedding_duration, 3),
    'similar_questions_duration': round(similar_questions_duration, 3),
    'table_matching_duration': round(table_matching_duration, 3),
    'sql_generation_duration': round(sql_generation_duration, 3),
    'bq_validation_duration': round(bq_validation_duration, 3),
    'bq_execution_duration': round(bq_execution_duration, 3),
    'sql_added_to_vector_db_duration' : round(sql_added_to_vector_db_duration, 3)
  }

  return response

def execute_final_sql(generated_sql):
  df = pandas_gbq.read_gbq(generated_sql, project_id=PROJECT_ID)
  return df

logger.info("-------------------------------------------------------------------------------")
logger.info("-------------------------------------------------------------------------------")

#vertexai.init(project=PROJECT_ID, location="us-central1")
model=createModel(PROJECT_ID, "us-central1",model_id)
chat_model=createModel(PROJECT_ID, REGION,chat_model_id)

# Run the SQL commands now.
asyncio.run(pgvector_handler.init_pgvector_conn())  # type: ignore

logger.info("Starting nl2sql module")

init_table_and_columns_desc()

response = call_gen_sql("What is the brand with the most purchases in the last year?")
logger.info('Answer:\n' + json.dumps(response, indent=2))