# Common Imports
import datetime
from datetime import datetime, timezone
import hashlib
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd
from pandas import DataFrame
from google.cloud.exceptions import NotFound
from logging import exception
import asyncio
import asyncpg
from google.cloud.sql.connector import Connector
from pgvector.asyncpg import register_vector
from vertexai.language_models import TextEmbeddingModel
import logging

# create logger
logger = logging.getLogger('pgvector_handler')

source_type='BigQuery'

# @markdown Provide the below details to start using the notebook
PROJECT_ID='cdp-demo-flocquet'
REGION = 'europe-west1'
DATAPROJECT_ID='cdp-demo-flocquet'
AUTH_USER='admin@fabienlocquet.altostrat.com'

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
ENABLE_ANALYTICS=True
DATASET_NAME='nl2sql'
DATASET_LOCATION='EU'
LOG_TABLE_NAME='query_logs'

# PGVECTOR (Cloud SQL Postgres) Info.
database_password = "hr_tutorial"
instance_name = "pg15-nl2sql-pgvector"
database_name = "nl2sql-admin"
database_user = "nl2sql-admin"
num_table_matches = 5
num_column_matches = 20
similarity_threshold = 0.1
num_sql_matches=3

# @markdown Create an HNSW index
m =  24
ef_construction = 100
operator =  "vector_cosine_ops"  # ["vector_cosine_ops", "vector_l2_ops", "vector_ip_ops"]

# Palm Models to use
embeddings_model='textembedding-gecko@003'

def init_pgvector_handler(query: str, schema: str = 'userbase', group_concat_max_len: int = 102400):


    # initialize Connector object
    connector = Connector()

    # function to return the database connection object
    def getconn():
        conn = connector.connect(
            PROJECT_ID+':'+REGION+':'+instance_name,
            "pg8000",
            user=database_user,
            password=database_password,
            db=database_name
        )
        return conn

    # create connection pool with 'creator' argument to our connection object function
    pool = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    # connect to connection pool
    with pool.connect() as db_conn:
        # query and fetch ratings table
        results = db_conn.execute(text(query))
        db_conn.commit()
        db_conn.close()

    return 'SQL executed successsfully'

# Configure pgVector as the Vector Store
def text_embedding(question):
    """Text embedding with a Large Language Model."""
    model = TextEmbeddingModel.from_pretrained(embeddings_model)
    embeddings = model.get_embeddings([question])
    for embedding in embeddings:
        vector = embedding.values
        logger.debug("Length of Embedding Vector: " + str(len(vector)))
    return vector

def add_table_desc_2_pgvector(table_comments_df):

  epoch_time = datetime.now(timezone.utc)

  requestor=str(AUTH_USER)
  for index, row in table_comments_df.iterrows():
    embeddings=[]
    # Define ID ... hashed value of the question+requestor+schema
    q_value=str(DATAPROJECT_ID) + '.' + str(source_type) + '.' + str(requestor)+ '.' + str(row['owner']) + '.' + str(row['table_name']) + '.' + str(row['detailed_description'])
    hash_object = hashlib.md5(str(q_value).encode())
    hex_dig = hash_object.hexdigest()
    idx=str(hex_dig)

    embeddings=text_embedding(str(row['detailed_description']))

    sql=f'''
      insert into table_embeddings(id,detailed_description,requestor,table_catalog,table_schema,table_name,added_epoch,source_type,embedding)
      values(\'{idx}\',
      \'{str(row['detailed_description']).replace("'","''")}\',
      \'{str(requestor).replace("'","''")}\',
      \'{str(DATAPROJECT_ID).replace("'","''")}\',
      \'{str(row['owner']).replace("'","''")}\',
      \'{str(row['table_name']).replace("'","''")}\',
      \'{epoch_time}\',
      \'{source_type}\',
      \'{embeddings}\')
    '''
    logger.info("Adding table description to pgVector DB: " + q_value)
    logger.debug("SQL: " + sql)
    ret=init_pgvector_handler(sql)
  return "Table Description added to the vector DB"

def pgvector_table_desc_exists(df: DataFrame):

    requestor=str(AUTH_USER)
    uninitialized_tables = []
    for index, row in df.iterrows():

        table_sql=f'''
        select detailed_description from table_embeddings
        where
            requestor=\'{str(requestor)}\' and
            table_catalog=\'{str(DATAPROJECT_ID)}\' and
            table_schema=\'{str(row['owner'])}\' and
            table_name=\'{str(row['table_name'])}\'
        '''

        logger.info("Checking whether table " + str(DATAPROJECT_ID) + "." + str(row['owner']) + "." + str(row['table_name']) + " exists in pgVector DB")
        logger.debug("SQL: " + table_sql)

        table_results_joined = pgvector_get_data(table_sql)

        if len(table_results_joined) == 0:
            logger.info("Table '" + str(row['table_name']) + "' not present in pgVector.")
            uninitialized_tables.append(row['table_name'])
        else:
            logger.info("Table '" + str(row['table_name']) + "' already present in pgVector")
        
    return uninitialized_tables

def pgvector_get_data(sql: str):
    from sqlalchemy.sql import text

    epoch_time = datetime.now(timezone.utc)

    matches = []

    # initialize Connector object
    connector = Connector()

    # function to return the database connection object
    def getconn():
        conn = connector.connect(
            PROJECT_ID+':'+REGION+':'+instance_name,
            "pg8000",
            user=database_user,
            password=database_password,
            db=database_name
        )
        return conn

    # create connection pool with 'creator' argument to our connection object function
    pool = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    table_results_joined = ''

    with pool.connect() as db_conn:

        table_results = db_conn.execute(text(sql)).fetchall()
        for r in table_results:
            table_results_joined = table_results_joined + r[0] + '\n'
        db_conn.close()

    return table_results_joined

def add_column_desc_2_pgvector(columns_df):

  from datetime import datetime, timezone
  import hashlib
  import time
  epoch_time = datetime.now(timezone.utc)

  requestor=str(AUTH_USER)
  for index, row in columns_df.iterrows():
    embeddings=[]
    # Define ID ... hashed value of the question+requestor+schema
    q_value=str(DATAPROJECT_ID) + '.' + str(source_type) + '.' + str(requestor) + '.' + str(row['owner']) + '.' + str(row['table_name']) + '.' + str(row['column_name']) + '.' + str(row['detailed_description'])
    hash_object = hashlib.md5(str(q_value).encode())
    hex_dig = hash_object.hexdigest()
    idx=str(hex_dig)


    embeddings=text_embedding(str(row['detailed_description']))

    sql=f'''
      insert into column_embeddings(id,detailed_description,requestor,table_catalog,table_schema,table_name,column_name,added_epoch,source_type,embedding)
      values(\'{idx}\',
      \'{str(row['detailed_description']).replace("'","''")}\',
      \'{str(requestor).replace("'","''")}\',
      \'{str(DATAPROJECT_ID).replace("'","''")}\',
      \'{str(row['owner']).replace("'","''")}\',
      \'{str(row['table_name']).replace("'","''")}\',
      \'{str(row['column_name']).replace("'","''")}\',
      \'{epoch_time}\',
      \'{source_type}\',
      \'{embeddings}\')
    '''

    logger.info("Adding column " + row['column_name'] + " to table entry " + str(DATAPROJECT_ID) + '.' + str(row['owner']) + '.' + str(row['table_name']) + " in pgVector DB")
    logger.debug("SQL: " + sql)
    ret=init_pgvector_handler(sql)

  return "Columns records added to Vector DB"


def get_tables_ann_pgvector(question: str, query: str, group_concat_max_len: int = 102400):
    from sqlalchemy.sql import text

    embedding_resp=text_embedding(question)
    matches = []
    similarity_threshold = 0.1
    num_matches = 5


    # initialize Connector object
    connector = Connector()

    # function to return the database connection object
    def getconn():
        conn = connector.connect(
            PROJECT_ID+':'+REGION+':'+instance_name,
            "pg8000",
            user=database_user,
            password=database_password,
            db=database_name
        )
        return conn

    # create connection pool with 'creator' argument to our connection object function
    pool = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    sql=f'''
        WITH vector_matches AS (
                              SELECT id, table_schema, table_name,
                              detailed_description,
                              1 - (embedding <=> \'{embedding_resp}\') AS similarity
                              FROM table_embeddings
                              WHERE 1 - (embedding <=> \'{embedding_resp}\') > {similarity_threshold}
                              ORDER BY similarity DESC
                              LIMIT {num_matches}
                            )
                            SELECT id, detailed_description, similarity
                            FROM vector_matches
                            where table_schema=\'{schema}\'
                            '''

    logger.debug("SQL: " + sql)
    # connect to connection pool
    with pool.connect() as db_conn:
        results = db_conn.execute(text(sql)).fetchall()

        if len(results) == 0:
            raise Exception("Did not find any results. Adjust the query parameters.")

        for r in results:
            logger.info(r[0])
            # Collect the description for all the matched similar toy products.
            matches.append(
                {
                    "id": r[0],
                    "description": r[1],
                    "similarity": r[2]
                }
            )

        matches = pd.DataFrame(matches)
        db_conn.close()

    return matches


#question = "Display the result of selecting test word from dual"
#res=search_sql_vector_by_id('HR',question,'Y')
#logger.info(res)

def search_sql_vector_by_id(schema, question, valid):
    from sqlalchemy.sql import text
    msg=''

    # Define ID ... hashed value of the question
    q_value=str(PROJECT_ID) + '.' + str(source_type) + '.' + str(AUTH_USER) + '.' + str(schema) + '.' + str(question) + '.' + str(valid)
    hash_object = hashlib.md5(str(q_value).encode())
    hex_dig = hash_object.hexdigest()
    idx=str(hex_dig)


    # initialize Connector object
    connector = Connector()

    # function to return the database connection object
    def getconn():
        conn = connector.connect(
            PROJECT_ID+':'+REGION+':'+instance_name,
            "pg8000",
            user=database_user,
            password=database_password,
            db=database_name
        )
        return conn

    # create connection pool with 'creator' argument to our connection object function
    pool = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    sql=f'''
      select generated_sql from sql_embeddings
      WHERE
        id = \'{idx}\' and requestor=\'{AUTH_USER}\'

    '''

    #logger.info(sql)
    # connect to connection pool
    with pool.connect() as db_conn:
        results = db_conn.execute(text(sql)).fetchall()

        if len(results) == 0:
            logger.error("SQL not found in Vector DB")
            msg='SQL Not Found in Vector DB'

        for r in results:
            logger.info('Record found in Vector DB. Parameters:')
            #logger.info(r[0])
            msg=str(r[0])

        db_conn.close()

    return msg

#question = "Display the result of selecting test word from dual"
#res=search_sql_vector_by_id('HR',question,'Y')
#logger.info(res)



def search_sql_nearest_vector(schema, question, question_text_embedding, valid):
    from sqlalchemy.sql import text

    msg='Examples:\n'

    matches = []
    #similarity_threshold = 0.1
    #num_matches = 3


    # initialize Connector object
    connector = Connector()

    # function to return the database connection object
    def getconn():
        conn = connector.connect(
            PROJECT_ID+':'+REGION+':'+instance_name,
            "pg8000",
            user=database_user,
            password=database_password,
            db=database_name
        )
        return conn

    # create connection pool with 'creator' argument to our connection object function
    pool = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    sql=f'''
        WITH vector_matches AS (
                              SELECT id, question, generated_sql, requestor, table_schema, table_catalog,
                              1 - (embedding <=> \'{question_text_embedding}\') AS similarity
                              FROM sql_embeddings
                              WHERE 1 - (embedding <=> \'{question_text_embedding}\') > {similarity_threshold}
                              ORDER BY similarity DESC
                              LIMIT {num_sql_matches}
                            )
                            SELECT id, question, generated_sql, similarity
                            FROM vector_matches
                            where table_schema=\'{schema}\'
                            and requestor=\'{AUTH_USER}\'
                            and table_catalog=\'{PROJECT_ID}\'
                            '''

    with pool.connect() as db_conn:

        results = db_conn.execute(text(sql)).fetchall()

        if len(results) == 0:
            msg=''
            logger.info('No record near the query was found in the Vector DB.')

        for r in results:
            msg= msg + '\nQuestion:' + r[1] + '\n' + 'Generated SQL:' + r[2] + '\n'

        db_conn.close()

    return msg


# question="Display the employee name and city when employee ID is 103."
# ret=search_sql_nearest_vector(schema, question, 'Y')
# logger.info( ret )


def pgvector_handler(query: str, schema: str = 'userbase', group_concat_max_len: int = 102400):


    # initialize Connector object
    connector = Connector()

    # function to return the database connection object
    def getconn():
        conn = connector.connect(
            PROJECT_ID+':'+REGION+':'+instance_name,
            "pg8000",
            user=database_user,
            password=database_password,
            db=database_name
        )
        return conn

    # create connection pool with 'creator' argument to our connection object function
    pool = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    # connect to connection pool
    with pool.connect() as db_conn:
        # query and fetch ratings table
        results = db_conn.execute(text(query))
        db_conn.commit()
        db_conn.close()

    return 'SQL executed successsfully'



def add_vector_sql_collection(schema, question, final_sql, question_text_embedding, valid):
  from datetime import datetime, timezone
  import hashlib
  import time

  epoch_time = datetime.now(timezone.utc)

  requestor=str(AUTH_USER)

  # Define ID ... hashed value of the question+requestor+schema
  q_value=str(PROJECT_ID) + '.' + str(source_type) + '.' + str(requestor) + '.' + str(schema) + '.' + str(question) + '.' + str(valid)
  hash_object = hashlib.md5(str(q_value).encode())
  hex_dig = hash_object.hexdigest()
  idx=str(hex_dig)


  sql=f'''
        insert into sql_embeddings
        (id,question,generated_sql,requestor,table_catalog,table_schema,added_epoch,source_type,embedding)
        values(\'{idx}\',
        \'{question.replace("'","''")}\',
        \'{final_sql.replace("'","''")} \',
        \'{str(requestor).replace("'","''")}\',
        \'{str(PROJECT_ID).replace("'","''")}\',
        \'{schema}\',
        \'{epoch_time}\',
        \'{source_type}\',
        \'{question_text_embedding}\')
        on conflict do nothing;
    '''

  ret=pgvector_handler(sql)

  return 'Question ' + str(question) + ' added to the Vector DB'



def get_tables_colums_vector(question, question_text_embedding):
    from sqlalchemy.sql import text

    table_results_joined=""
    column_results_joined=""

    matches = []

    # initialize Connector object
    connector = Connector()

    # function to return the database connection object
    def getconn():
        conn = connector.connect(
            PROJECT_ID+':'+REGION+':'+instance_name,
            "pg8000",
            user=database_user,
            password=database_password,
            db=database_name
        )
        return conn

    # create connection pool with 'creator' argument to our connection object function
    pool = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    table_sql=f'''
        WITH vector_matches AS (
                              SELECT id, table_schema, table_name, table_catalog,
                              detailed_description, requestor,
                              1 - (embedding <=> \'{question_text_embedding}\') AS similarity
                              FROM table_embeddings
                              WHERE 1 - (embedding <=> \'{question_text_embedding}\') > {similarity_threshold}
                              ORDER BY similarity DESC
                              LIMIT {num_table_matches}
                            )
                            SELECT id, detailed_description, similarity
                            FROM vector_matches
                            where table_schema=\'{schema}\'
                            and requestor=\'{AUTH_USER}\'
                            and table_catalog=\'{DATAPROJECT_ID}\'
                            '''

    with pool.connect() as db_conn:

        table_results = db_conn.execute(text(table_sql)).fetchall()
        for r in table_results:
            table_results_joined = table_results_joined + r[1] + '\n'

        db_conn.close()

    return table_results_joined


# Configure pgVector extension if the same does not exist
async def init_pgvector_conn():
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{PROJECT_ID}:{REGION}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await register_vector(conn)

        await conn.close()