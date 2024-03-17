from google.cloud import bigquery
import logging.config
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation
from jinja2 import Template
from datetime import date, timedelta
from google.api_core.exceptions import Conflict
import subprocess

# Default values
base_path = Path.cwd() / "installation_scripts" / "bigquery_dataset"
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(base_path / "config.ini")

default_config = config["DEFAULT"]
project_id = default_config['project_id']
dataset_id = default_config['dataset_id']
region = default_config['region']
users_percentage = default_config['users_percentage']


data = {
    'project_id': project_id,
    'dataset_id': dataset_id,
    'events_start_date': default_config['events_start_date'],
    'events_end_date': date.today() - timedelta(days = 1),
    'users_percentage': default_config['users_percentage'],
    'max_sessions_per_user': default_config['max_sessions_per_user']
}

users_schema_path = base_path / "users_schema.json"
products_schema_path = base_path / "products_schema.json"
sql_file_path = base_path / "bigquery_demo_dataset.sql"

try:
    subprocess.run(["gcloud", "services", "enable", "bigquery.googleapis.com", "--project", project_id], check=True)
    print("BigQuery API has been enabled.")

except subprocess.CalledProcessError:
    print("Failed to enable BigQuery API. Please check your gcloud configuration and permissions.")

# Enable BigQuery Client
client = bigquery.Client(project=project_id, location=region)

# Setup of the Dataset
dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
dataset = bigquery.Dataset(dataset_ref)

try:
    client.create_dataset(dataset)
    print(f"Dataset {dataset_id} created with success.")
except Conflict:
    print(f"Dataset {dataset_id} already exists.")

#---------- Create Users table ----------
uri = "gs://genai-cdp-demo/users/users-*.csv"
table_id = project_id + '.' + dataset_id + '.users'

# To load a schema file use the schema_from_json method.
schema = client.schema_from_json(users_schema_path)

job_config = bigquery.LoadJobConfig(
    # To use the schema you loaded pass it into the
    # LoadJobConfig constructor.
    schema=schema,
    skip_leading_rows=1,
)

# Pass the job_config object to the load_table_from_file,
# load_table_from_json, or load_table_from_uri method
# to use the schema on a new table.
load_job = client.load_table_from_uri(
    uri, table_id, job_config=job_config
)  # Make an API request.

load_job.result()  # Waits for the job to complete.

#---------- Create Products table ----------
uri = "gs://genai-cdp-demo/products/products.csv"
table_id = project_id + '.' + dataset_id + '.products'

# To load a schema file use the schema_from_json method.
schema = client.schema_from_json(products_schema_path)

job_config = bigquery.LoadJobConfig(
    # To use the schema you loaded pass it into the
    # LoadJobConfig constructor.
    schema=schema,
    skip_leading_rows=1,
)

# Pass the job_config object to the load_table_from_file,
# load_table_from_json, or load_table_from_uri method
# to use the schema on a new table.
load_job = client.load_table_from_uri(
    uri, table_id, job_config=job_config
)  # Make an API request.

load_job.result()  # Waits for the job to complete.

# Reading SQL script from the file
with open(sql_file_path, "r") as sql_file:
    sql_script_tmpl = sql_file.read()
    j2_template = Template(sql_script_tmpl)
    sql_script = j2_template.render(data)

with open("generated.sql", "w") as text_file:
    text_file.write(sql_script)

# Split the script in multiples commands if needed
sql_commands = sql_script.split(';')[:-1]

# Run every SQL commands in the file
# for command in sql_commands:
#     if command.strip():  # Check if the command is not empty
#         query_job = client.query(command, location=region)
#         query_job.result()  # Wait for the end of the request
#         print("SQL Command in success.")

query_job = client.query(sql_commands, location=region)
query_job.result()  # Wait for the end of the request

print("All the SQL commands in success.")