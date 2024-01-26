# Common Imports
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import CodeGenerationModel, ChatModel, CodeChatModel, TextGenerationModel
import pandas_gbq
import pgvector_handler
import os, logging, json, time, yaml
from json import JSONDecodeError
import cfg
import chat_session
import bigquery_handler
import concurrent.futures

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
  global sql_generation_model
  sql_generation_model = createModel(cfg.project_id, "us-central1", cfg.sql_generation_model_id)

  global validation_model
  validation_model = createModel(cfg.project_id, "us-central1", cfg.validation_model_id)

  global chat_model
  chat_model = createModel(cfg.project_id, cfg.region, cfg.chat_model_id)

  if cfg.update_db_at_startup is True:
    init_table_and_columns_desc()

# Initialize Palm Models to be used
def createModel(PROJECT_ID, REGION, model_id):

  if model_id == 'code-bison-32k':
    model = CodeGenerationModel.from_pretrained(model_id)
  elif model_id == 'gemini-pro':
    model = GenerativeModel(model_id)
  elif model_id == 'codechat-bison-32k':
    model = CodeChatModel.from_pretrained(model_id)
  elif model_id == 'chat-bison-32k':
    model = ChatModel.from_pretrained(model_id)
  elif model_id == 'text-unicorn':
    model = TextGenerationModel.from_pretrained(model_id)
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
    insert_sample_queries_lookup(tables_list)


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


def generate_sql(model, context_prompt):
  if isinstance(model, GenerativeModel):
    generated_sql_json = model.generate_content(
      context_prompt,
      generation_config={
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_p": 1
    })
    generated_sql = generated_sql_json.candidates[0].content.parts[0].text
  elif isinstance(model, TextGenerationModel):
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_k": 40
    }
    generated_sql_json = model.predict(
      context_prompt,
      **parameters)
    generated_sql = generated_sql_json.text
  return chat_session.clean_json(generated_sql)


def gen_dyn_rag_sql(question,table_result_joined, similar_questions):

  similar_questions_str = chat_session.question_to_query_examples(similar_questions)

  not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'
  context_prompt = f"""
You are a BigQuery SQL guru. Write a SQL conformant query for Bigquery that answers the following question while using the provided context to correctly refer to the BigQuery tables and the needed column names.

Guidelines:
{cfg.prompt_guidelines}

Tables Schema:
{table_result_joined}

{similar_questions_str}

[Question]:
{question}

[SQL Generated]:

    """

    #Column Descriptions:
    #{column_result_joined}

  logger.debug('LLM GEN SQL Prompt: \n' + context_prompt)

  context_query = generate_sql(sql_generation_model, context_prompt)

  return context_query

def sql_explain(question, generated_sql, table_schema):
  context_prompt = f"""
You are a BigQuery SQL guru. Generate a high-level question to which the [SQL Query] answers.

Guidelines:
  - Analyze the database and the table schema provided as parameters and understand the relations (column and table relations) and the column descriptions.
  - In the generated question, stay as concise as possible while not missing any filtering and time range specified by the [SQL query].
  - In the generated question, if no time range is specified for a specific filter, consider that it is global, or total.

[Tables Schema]:
{table_schema}

[SQL Query]:
{generated_sql}
"""

  logger.debug('Validation - Question Generation from SQL Prompt: \n' + context_prompt)

  generated_question = generate_sql(validation_model, context_prompt)

  response_json = {
    "is_matching": "{Answer with 'True' or 'False' depending on the outcome of the comparison between the provided Query and the Reference Question}",
    "mismatch_details": "{Write all identified missing or incorrect filters from the Query. If not, return an empty string. Be specific when highlighting a difference}"
  }

  context_prompt = f"""
Compare a Query to a Reference Question and assess whether they are equivalent or not.

[Guidelines]:
- Answer using the following json format:
{response_json}
- Remove ```json prefix and ``` suffix from the outputs.
- Use double quotes "" for json property names and values in the returned json object.

[Reference Question]:
{question}

[Query]:
{generated_question}
"""

  logger.debug('Validation - Question Comparison Prompt: \n' + context_prompt)

  try:

    sql_explanation = generate_sql(validation_model, context_prompt)
    logger.info("Validation status: \n" + sql_explanation)
    validation_json = json.loads(sql_explanation, strict=False)

  except JSONDecodeError as e:
    logger.error("Error while deconding JSON response:: " + str(e))
    sql_explanation['is_matching'] = 'False'
    sql_explanation['mismatch_details'] = 'Returned JSON malformed'
  except Exception as e:
    logger.error("Exception: " + str(e))
    sql_explanation['is_matching'] = 'False'
    sql_explanation['mismatch_details'] = 'Undefined error. Retry'


  validation_json['generated_question'] = generated_question

  return validation_json


def call_gen_sql(question):

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
  }

  workflow = {
    'stop_loop': False,
    'error_retry_max_count': cfg.sql_max_error_retry,
    'explanation_retry_max_count': cfg.sql_max_explanation_retry,
    'error_retry_count': 0,
    'explanation_retry_count': 0
  }

  status = {
    'status': 'Error',
    'unrelated_question': False,
    'sql_generation_success': False,
    'sql_validation_success': False,
    'error_message': None
  }

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
    similar_questions_return = pgvector_handler.search_sql_nearest_vector(cfg.schema, question, question_text_embedding, 'Y')
    metrics['similar_questions_duration'] = time.time() - start_time
    logger.info("Found similar questions:\n" + str(similar_questions_return))

    error_correction_chat_session=None
    explanation_correction_chat_session=None
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
        generated_sql=gen_dyn_rag_sql(question,table_result_joined, similar_questions_return)
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

      # Execute SQL test plan and semantic validation in parallel in order to minimize overall query generation time
      logger.info("Executing SQL test plan and SQL query semantic validation in parallel...")
      with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_test_plan = executor.submit(bigquery_handler.execute_bq_query, generated_sql, dry_run=True)
        future_sql_validation = executor.submit(sql_explain, question, generated_sql, table_result_joined)
        status['bq_status'],_ = future_test_plan.result()
        sql_explanation = future_sql_validation.result()

      if status['bq_status']['status'] == 'Success':

        status['sql_generation_success'] = True
        if sql_explanation['is_matching'] == 'True':
          status['sql_validation_success'] = True
        else:
          status['sql_validation_success'] = False       
        workflow['stop_loop'] = True

      else:  # Failure on BigQuery SQL execution
          
          logger.info("Requesting SQL rewrite using chat LLM. Retry number #" + str(workflow['error_retry_count']))
          append_2_bq_result = bigquery_handler.append_2_bq(
            cfg.sql_generation_model_id,
            question,
            generated_sql, 'N', 'Y',
            'explain_plan_validation',
            status['bq_status']['error_message'] )
          ### Need to call retry
          if error_correction_chat_session is None: error_correction_chat_session = chat_session.SQLCorrectionChat(sql_generation_model)
          rewrite_result = error_correction_chat_session.get_chat_response(
            question,
            generated_sql,
            table_result_joined,
            status['bq_status']['error_message'],
            similar_questions_return)
          generated_sql=rewrite_result

      if workflow['stop_loop'] is not True:
        workflow['error_retry_count'] += 1
        if workflow['error_retry_count'] > workflow['error_retry_max_count']:
          workflow['stop_loop'] = True
          status['error_message'] = 'Question cannot be answered using the configured datasets'

    # After the loop, check whether is successfully generated a valid SQL query or not
    if status['sql_generation_success'] is False:

      # If the generation was unsuccessful, set the status flags accordingly
      if workflow['error_retry_count'] > workflow['error_retry_max_count']:
        logger.info('Oopss!!! Could not find a valid SQL. This is the best I came up with !!!!!\n' + generated_sql)
        workflow['sql_generation_success'] = False

      # If query is unrelated to the dataset
      if status['unrelated_question'] is True:
        logger.info('Question cannot be answered using this dataset!')
        status['sql_generation_success'] = False
        status['error_message'] = 'Question cannot be answered using the configured datasets'
        append_2_bq_result = bigquery_handler.append_2_bq(
          cfg.model_id,
          question,
          'Question cannot be answered using this dataset!',
          'N', 'N', 'unrelated_question', '')
    
    # If SQL query was successfully generated and tested on BigQuery, proceed with query validation
    else:

      workflow['stop_loop'] = status['sql_validation_success']

      while workflow['stop_loop'] is False:
      
        logger.info("Rewriting SQL request to address previous generated SQL semantic issues...")

        if explanation_correction_chat_session is None: explanation_correction_chat_session = chat_session.ExplanationCorrectionChat(sql_generation_model)

        generated_sql = explanation_correction_chat_session.get_chat_response(
          table_result_joined,
          similar_questions_return,
          question,
          generated_sql,
          sql_explanation['generated_question'],
          sql_explanation['mismatch_details'])
        
        logger.info("New SQL request generated")

        # Check whether the generated query actually answers the initial question by invoking GenAI
        # to analyze what the generated SQL is doing. It returns a json containing the result of the analysis
        logger.info("Check whether the rewritten SQL query now matches the original question...")
        sql_explanation = sql_explain(question, generated_sql, table_result_joined)
        logger.debug('Generated SQL explanation: ' + json.dumps(sql_explanation))
        if 'is_matching' in sql_explanation and sql_explanation['is_matching'] == 'True':
          logger.info("Generated SQL explanation matches initial question.")
          status['sql_validation_success'] = True

        if status['sql_validation_success'] is True:
        
          # IF required, final validation of the rewritten query by executing a dry-run on BigQuery
          logger.info('Executing BigQuery dry-run for the validated rewritten SQL Query, only if the initial validation failed...')
          status['bq_status'], sql_result_df = bigquery_handler.execute_bq_query(generated_sql, dry_run=True)

          if status['bq_status']['status'] == 'Success':
            workflow['stop_loop'] = True
            logger.info('BigQuery dry-run successfully completed for rewritten SQL Query.')
          else:
            logger.error('BigQuery dry-run failed for the rewritten SQL Query. Can\'t answer user query.')
            workflow['stop_loop'] = False
            status['sql_validation_success'] = False
            sql_explanation['mismatch_details'] = status['bq_status']['error_message']

        workflow['explanation_retry_count'] += 1
        if workflow['explanation_retry_count'] > workflow['explanation_retry_max_count']:
          workflow['stop_loop'] = True

      #appen_2_bq_result = append_2_bq(cfg.validation_model_id, question, generated_sql, 'N', 'N', '', '')


    if status['sql_validation_success'] is True:

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

    else:
      logger.info("Can't find a correct SQL Query that matches exactly the initial questions.")
      status['status'] = 'Error'
      status['error_message'] = sql_explanation['mismatch_details']
      
    if status['status'] == 'Success' and cfg.auto_add_knowngood_sql is True:
      #### Adding to the Known Good SQL Vector DB
      logger.info("Adding Known Good SQL to Vector DB...")
      start_time = time.time()
      pgvector_handler.add_vector_sql_collection(cfg.schema, question, generated_sql, question_text_embedding, 'Y')
      sql_added_to_vector_db_duration = time.time() - start_time
      status['sql_added_to_vector_db_duration'] = sql_added_to_vector_db_duration
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
        sql_result_str = "Audience Size: " + str(sql_result_df.iat[0,0].item())
      else:
        sql_result_str = "No users matches the question :("
      is_audience_result = True
    else:
      sql_result_str = sql_result_df.to_html()
      is_audience_result = True
  else:
    sql_result_str = ''
    is_audience_result = False

  response = {
    'status': status['status'],
    'error_message': status['error_message'] if sql_result_df is None else None,
    'generated_sql': '<pre>' + generated_sql + '</pre>' if generated_sql != '' else None,
    'sql_result': sql_result_str if sql_result_df is not None else None,
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