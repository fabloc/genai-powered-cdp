import configparser, json
from configparser import ExtendedInterpolation

config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
config.read('config.ini')
gcp_config = config['GOOGLE_CLOUD']
tables_config = config['TABLES']
vector_config = config['VECTOR_DATABASE']
execution_config = config['EXECUTION']
models_config = config['ML_MODELS']
analytics_config = config['ANALYTICS']

# Google Cloud variables
project_id = gcp_config['project_id']
region = gcp_config['region']
dataproject_id = gcp_config['dataproject_id']
auth_user = gcp_config['auth_user']

# Tables variables
source_type = tables_config['source_type']
tables = tables_config['tables']
tables = json.loads(tables)
schema = tables_config['schema']
user_dataset = tables_config['user_dataset']

# Execution variables
sql_validation = execution_config['sql_validation']
inject_one_error = execution_config.getboolean('inject_one_error')
sql_max_error_retry = execution_config.getint('sql_max_error_retry')
sql_max_explanation_retry = execution_config.getint('sql_max_explanation_retry')
auto_add_knowngood_sql = execution_config.getboolean('auto_add_knowngood_sql')
execute_final_sql = execution_config.getboolean('execute_final_sql')
display_bq_max_results = execution_config.getint('display_bq_max_results') if 'display_bq_max_results' in execution_config else 100

# Analytics variables
enable_analytics = analytics_config.getboolean('enable_analytics')
dataset_name = analytics_config['dataset_name']
dataset_location = analytics_config['dataset_location']
log_table_name = analytics_config['log_table_name']

# ML Models variables
fast_sql_generation_model = models_config['fast_sql_generation_model_id']
fine_sql_generation_model = models_config['fine_sql_generation_model_id']
sql_correction_model_id = models_config['sql_correction_model_id']
validation_model_id = models_config['validation_model_id']
embeddings_model = models_config['embeddings_model']
models_timeout = models_config.getint('models_timeout') if 'models_timeout' in models_config else 20

# Vector DB variables
update_db_at_startup = vector_config.getboolean('update_db_at_startup')
database_password = vector_config['database_password']
instance_name = vector_config['instance_name']
database_name = vector_config['database_name']
database_user = vector_config['database_user']
num_table_matches = vector_config.getint('num_table_matches')
num_column_matches = vector_config.getint('num_column_matches')
similarity_threshold = vector_config.getfloat('similarity_threshold')
num_sql_matches = vector_config.getint('num_sql_matches')

# @markdown Create an HNSW index
m =  vector_config.getint('m')
ef_construction = vector_config.getint('ef_construction')
operator =  vector_config['operator']  # ["vector_cosine_ops", "vector_l2_ops", "vector_ip_ops"]

# Prompt Configuration
not_related_msg='select \'Question is not related to the dataset\' as unrelated_answer from dual;'
prompt_guidelines = """- Only answer questions relevant to the tables listed in [Table Schema]. If a non-related question comes, answer exactly: 'Question is not related to any listed dataset'.
- Join as minimal tables as possible.
- When joining tables ensure all join columns are the same data_type.
- Analyze the database and the table schema provided as parameters and undestand the relations (column and table relations).
- For questions about finding a number of users, always approximate the number of users.
- For questions about listing brands, categories, products, always find distinct elements.
- Never use "user_id" in the "GROUP BY" statement for the top "SELECT" block.
- Never use the 'ARRAY_FILTER' function.
- Never use the 'DISTINCT_AGG' function.
- Convert TIMESTAMP to DATE.
- Consider alternative options to CAST function. If performing a CAST, use only Bigquery supported datatypes.
- Don't include any comments in code.
- Give human readable names to tables and columns in the generated SQL query, in lowercase.
- Remove "sql", ```sql and ``` strings from the output and generate the SQL in single line.
- Tables should be refered to using a fully qualified name (project_id.owner.table_name).
- Use all the non-aggregated columns from the "SELECT" statement while framing "GROUP BY" block.
- Return syntactically and semantically correct SQL for BigQuery with proper relation mapping i.e project_id, owner, table and column relation.
- Use ONLY the column names (column_name) mentioned in Table Schema. DO NOT USE any other column names outside of this.
- Associate column_name mentioned in Table Schema only to the table_name specified under [Table Schema].
- Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed.
- Table names are case sensitive. DO NOT uppercase or lowercase the table names.
- Owner (dataset) is case sensitive. DO NOT uppercase or lowercase the owner.
- Project_id is case sensitive. DO NOT uppercase or lowercase the project_id."""