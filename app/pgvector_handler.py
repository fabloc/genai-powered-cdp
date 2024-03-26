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
from google.cloud.sql.connector import IPTypes
from pgvector.asyncpg import register_vector
import logging
import cfg

class VectorConnectionMgr:

    # Vector DB connection objects
    db_connector = None
    db_connection = None

    def __init__(self, logger):

        # create logger
        self.logger = logger

        if self.db_connection is None:

            # initialize Connector object
            db_connector = Connector()

            # create connection pool with 'creator' argument to our connection object function
            connection_pool = create_engine(
                "postgresql+pg8000://",
                creator=self.getconn,
            )

            self.db_connection = connection_pool.connect()

    # function to return the database connection object
    def getconn(self):
        if self.db_connector is None:
            self.db_connector = Connector()
        conn = self.db_connector.connect(
            cfg.project_id + ':' + cfg.region + ':' + cfg.instance_name,
            "pg8000",
            user=cfg.database_user,
            password=cfg.database_password,
            db=cfg.database_name,
            ip_type=cfg.sql_ip_type
        )
        return conn


    def execute_query(self, query: str, write=False, schema: str = 'userbase', group_concat_max_len: int = 102400):

        # connect to connection pool
        # query and fetch ratings table
        results = self.db_connection.execute(text(query))
        if write:
            self.db_connection.commit()
            return
        else:
            return results.fetchall()
    
    def close_connection(self):
        self.db_connection.close()


def shutdown():
    connection_mgr.close_connection()

# Configure pgVector extension if the same does not exist
async def add_pgvector_ext_async():
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            instance_connection_string = cfg.project_id + ":" + cfg.region + ":" + cfg.instance_name,  # Cloud SQL instance connection name
            driver = "asyncpg",
            user=cfg.database_user,
            password=cfg.database_password,
            db=cfg.database_name,
            ip_type=cfg.sql_ip_type
        )

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await register_vector(conn)

        await conn.close()

def add_pgvector_extension():
    asyncio.run(add_pgvector_ext_async())

# Creating tables 'table_embeddings' and 'sql_embeddings' if they do not exist
def create_tables():

    sql="""CREATE TABLE IF NOT EXISTS table_embeddings(
        id varchar(256) PRIMARY KEY,
        detailed_description TEXT,
        requestor varchar(50),
        table_catalog varchar(128),
        table_schema varchar(128),
        table_name varchar(50),
        added_epoch timestamp with time zone,
        source_type varchar(50),
        embedding vector(768))"""

    connection_mgr.execute_query(sql, write=True)

    sql="""CREATE TABLE IF NOT EXISTS sql_embeddings(
        id varchar(256) PRIMARY KEY,
        question TEXT,
        generated_sql TEXT,
        requestor varchar(50),
        table_catalog varchar(128),
        table_schema varchar(128),
        table_name varchar(50),
        added_epoch timestamp with time zone,
        source_type varchar(50),
        embedding vector(768))"""

    connection_mgr.execute_query(sql, write=True)

def create_indexes():
    # Create an HNSW index on the `table_embeddings` table.
    sql = f"""CREATE INDEX ON table_embeddings
    USING hnsw(embedding {cfg.operator})
    WITH (m = {cfg.m}, ef_construction = {cfg.ef_construction})
    """
    connection_mgr.execute_query(sql, write=True)

    # Create an HNSW index on the `sql_embeddings` table.
    sql = f"""CREATE INDEX ON sql_embeddings
    USING hnsw(embedding {cfg.operator})
    WITH (m = {cfg.m}, ef_construction = {cfg.ef_construction})
    """
    connection_mgr.execute_query(sql, write=True)

# Configure pgVector as the Vector Store
def text_embedding(question):
    from vertexai.language_models import TextEmbeddingModel
    """Text embedding with a Large Language Model."""
    model = TextEmbeddingModel.from_pretrained(cfg.embeddings_model)
    embeddings = model.get_embeddings([question])
    for embedding in embeddings:
        vector = embedding.values
        logger.debug("Length of Embedding Vector: " + str(len(vector)))
    return vector

def add_table_desc_2_pgvector(table_comments_df):

    epoch_time = datetime.now(timezone.utc)

    for index, row in table_comments_df.iterrows():
        embeddings=[]
        # Define ID ... hashed value of the question+requestor+schema
        q_value=str(cfg.dataproject_id) + '.' + str(cfg.source_type) + '.' + str(cfg.auth_user)+ '.' + str(row['owner']) + '.' + str(row['table_name']) + '.' + str(row['detailed_description'])
        hash_object = hashlib.md5(str(q_value).encode())
        hex_dig = hash_object.hexdigest()
        idx=str(hex_dig)

        embeddings = text_embedding(str(row['detailed_description']))

        sql=f'''
        insert into table_embeddings(id,detailed_description,requestor,table_catalog,table_schema,table_name,added_epoch,source_type,embedding)
        values(\'{idx}\',
        \'{str(row['detailed_description']).replace("'","''")}\',
        \'{str(cfg.auth_user).replace("'","''")}\',
        \'{str(cfg.dataproject_id).replace("'","''")}\',
        \'{str(row['owner']).replace("'","''")}\',
        \'{str(row['table_name']).replace("'","''")}\',
        \'{epoch_time}\',
        \'{cfg.source_type}\',
        \'{embeddings}\')
        '''
        logger.info("Adding table description to pgVector DB: " + q_value)
        logger.debug("SQL: " + sql)
        ret = connection_mgr.execute_query(sql, write=True)
    return "Table Description added to the vector DB"

def pgvector_table_desc_exists(df: DataFrame):

    uninitialized_tables = []
    for index, row in df.iterrows():

        table_sql=f'''
        select detailed_description from table_embeddings
        where
            requestor=\'{str(cfg.auth_user)}\' and
            table_catalog=\'{str(cfg.dataproject_id)}\' and
            table_schema=\'{str(row['owner'])}\' and
            table_name=\'{str(row['table_name'])}\'
        '''

        logger.info("Checking whether table " + str(cfg.dataproject_id) + "." + str(row['owner']) + "." + str(row['table_name']) + " exists in pgVector DB")
        logger.debug("SQL: " + table_sql)

        table_results_joined = connection_mgr.execute_query(table_sql)

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

    table_results_joined = ''

    table_results = connection_mgr.execute_query(sql)
    for r in table_results:
        table_results_joined = table_results_joined + r[0] + '\n'

    return table_results_joined


def get_tables_ann_pgvector(question: str, query: str, group_concat_max_len: int = 102400):
    from sqlalchemy.sql import text

    embedding_resp = text_embedding(question)
    matches = []
    similarity_threshold = 0.1
    num_matches = 5

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
            where table_schema=\'{cfg.dataset_id}\'
            '''

    logger.debug("SQL: " + sql)
    # connect to connection pool
    results = connection_mgr.execute_query(sql)

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

    return matches


#question = "Display the result of selecting test word from dual"
#res=search_sql_vector_by_id('HR',question,'Y')
#logger.info(res)

def search_sql_vector_by_id(schema, question, valid):
    from sqlalchemy.sql import text
    msg=''

    # Define ID ... hashed value of the question
    q_value=str(cfg.project_id) + '.' + str(cfg.source_type) + '.' + str(cfg.auth_user) + '.' + str(schema) + '.' + str(question) + '.' + str(valid)
    hash_object = hashlib.md5(str(q_value).encode())
    hex_dig = hash_object.hexdigest()
    idx=str(hex_dig)

    sql=f'''
        select generated_sql from sql_embeddings
        WHERE
            id = \'{idx}\' and requestor=\'{cfg.auth_user}\'

    '''

    #logger.info(sql)
    # connect to connection pool
    results = connection_mgr.execute_query(sql)

    if len(results) == 0:
        logger.error("SQL not found in Vector DB")
        msg='SQL Not Found in Vector DB'

    for r in results:
        logger.info('Record found in Vector DB. Parameters:')
        #logger.info(r[0])
        msg=str(r[0])

    return msg

#question = "Display the result of selecting test word from dual"
#res=search_sql_vector_by_id('HR',question,'Y')
#logger.info(res)


def search_sql_nearest_vector(question, question_text_embedding, valid):
    from sqlalchemy.sql import text

    msg='Examples:\n'

    matches = []
    #similarity_threshold = 0.1
    #num_matches = 3

    sql=f'''
        WITH vector_matches AS (
                SELECT id, question, generated_sql, requestor, table_schema, table_catalog,
                1 - (embedding <=> \'{question_text_embedding}\') AS similarity
                FROM sql_embeddings
                WHERE 1 - (embedding <=> \'{question_text_embedding}\') > {cfg.similarity_threshold}
                ORDER BY similarity DESC
                LIMIT {cfg.num_sql_matches}
            )
            SELECT id, question, generated_sql, similarity
            FROM vector_matches
            where table_schema=\'{cfg.dataset_id}\'
            and requestor=\'{cfg.auth_user}\'
            and table_catalog=\'{cfg.project_id}\'
            '''

    results = connection_mgr.execute_query(sql)

    if len(results) == 0:
        msg=''
        logger.info('No record near the query was found in the Vector DB.')

    query_examples = []
    for r in results:
        query_examples.append({
            "question": r[1],
            "sql_query": r[2]
        })

    return query_examples


# question="Display the employee name and city when employee ID is 103."
# ret=search_sql_nearest_vector(schema, question, 'Y')
# logger.info( ret )


def add_vector_sql_collection(question, final_sql, question_text_embedding, valid):
    from datetime import datetime, timezone
    import hashlib
    import time

    epoch_time = datetime.now(timezone.utc)

    requestor=str(cfg.auth_user)

    # Define ID ... hashed value of the question+requestor+schema
    q_value=str(cfg.project_id) + '.' + str(cfg.source_type) + '.' + str(requestor) + '.' + str(cfg.dataset_id) + '.' + str(question) + '.' + str(valid)
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
            \'{str(cfg.project_id).replace("'","''")}\',
            \'{cfg.dataset_id}\',
            \'{epoch_time}\',
            \'{cfg.source_type}\',
            \'{question_text_embedding}\')
            on conflict do nothing;
        '''

    connection_mgr.execute_query(sql, write=True)

    return 'Question ' + str(question) + ' added to the Vector DB'



def get_tables_colums_vector(question, question_text_embedding):
    from sqlalchemy.sql import text

    table_results_joined=""
    column_results_joined=""

    matches = []

    table_sql=f'''
        WITH vector_matches AS (
                            SELECT id, table_schema, table_name, table_catalog,
                            detailed_description, requestor,
                            1 - (embedding <=> \'{question_text_embedding}\') AS similarity
                            FROM table_embeddings
                            WHERE 1 - (embedding <=> \'{question_text_embedding}\') > {cfg.similarity_threshold}
                            ORDER BY similarity DESC
                            LIMIT {cfg.num_table_matches}
                            )
                            SELECT id, detailed_description, similarity
                            FROM vector_matches
                            where table_schema=\'{cfg.dataset_id}\'
                            and requestor=\'{cfg.auth_user}\'
                            and table_catalog=\'{cfg.dataproject_id}\'
                            '''

    table_results = connection_mgr.execute_query(table_sql)

    table_results_array = []
    for r in table_results:
        table_results_array.append(r[1])

    return '\n'.join(table_results_array)

# create logger
logger = logging.getLogger('pgvector_handler')

connection_mgr = VectorConnectionMgr(logger)