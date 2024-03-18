import pandas_gbq
from pathlib import Path
import pgvector_handler
import os, yaml, logging
from google.cloud.sql.connector import Connector
from google.cloud.sql.connector import IPTypes
from pgvector.asyncpg import register_vector
import asyncio
import cfg

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


def schema_generator(sql):
    formatted_sql = sql.format(**globals(), **locals())
    logger.info("BigQuery request: " + formatted_sql)
    df = pandas_gbq.read_gbq(formatted_sql, project_id=cfg.project_id, location=cfg.region, progress_bar_type=None)
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
        samples_filename_path = Path.cwd() / "config" / "queries_samples" / (table_name + '.yaml')
        if os.path.exists(samples_filename_path):
            with open(samples_filename_path) as stream:
                try:
                    table_queries_samples = yaml.safe_load(stream)
                    queries_samples.append(table_queries_samples)
                    for sql_query in queries_samples[0]:
                        question = sql_query['Question']
                        question_text_embedding = pgvector_handler.text_embedding(question)
                        pgvector_handler.add_vector_sql_collection(cfg.dataset_id, sql_query['Question'], sql_query['SQL Query'], question_text_embedding, 'Y')
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

    ln = ' \n  '
    aug_table_desc=f"""
  Table: `{cur_full_table}`
  Owner: {cur_table_owner}
  Columns:
  {ln.join(table_cols)}
  Primary Key: {final_pk_cols}
  Foreign Keys: {final_fk_cols}
  Project_id: {str(row_aug['project_id'])}
  Table Description: {str(row_aug['comments'])}"""

    # Works well
    aug_table_comments_df.at[index_aug, 'detailed_description'] = aug_table_desc
    logger.info("Table schema: \n" + aug_table_desc)
  return aug_table_comments_df


# create logger
logger = logging.getLogger('init_pgvector')

# Add the pgvector extension to the Cloud SQL instance
pgvector_handler.add_pgvector_extension()

# Create tables
pgvector_handler.create_tables()

# Create indexes on the tables
pgvector_handler.create_indexes()

# Build table schemas and populate the pgVector database
init_table_and_columns_desc()