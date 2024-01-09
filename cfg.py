import configparser
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
schema = tables_config['schema']
user_dataset = tables_config['user_dataset']

# Execution variables
sql_validation = execution_config['sql_validation']
inject_one_error = execution_config.getboolean('inject_one_error')
execute_final_sql = execution_config.getboolean('execute_final_sql')
sql_max_fix_retry = execution_config.getint('sql_max_fix_retry')
auto_add_knowngood_sql = execution_config.getboolean('auto_add_knowngood_sql')

# Analytics variables
enable_analytics = analytics_config.getboolean('enable_analytics')
dataset_name = analytics_config['dataset_name']
dataset_location = analytics_config['dataset_location']
log_table_name = analytics_config['log_table_name']

# ML Models variables
model_id = models_config['model_id']
chat_model_id = models_config['chat_model_id']
embeddings_model = models_config['embeddings_model']

# Vector DB variables
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