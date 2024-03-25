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
user_aggregates_schema_path = base_path / "user_aggregates_schema.json"
product_aggregates_schema_path = base_path / "product_aggregates_schema.json"
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
dataset.location = region
dataset.storage_billing_model = 'PHYSICAL'

try:
    client.create_dataset(dataset)
    print(f"Dataset {dataset_id} created with success.")
except Conflict:
    print(f"Dataset {dataset_id} already exists.")

#---------- Create Users table ----------
print("Importing Users table...")
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
print("Successfully imported Users table.")

#---------- Create Products table ----------
print("Importing Products table...")
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
print("Successfully imported Products table.")

# Create user_aggregates table using JSON schema
# To load a schema file use the schema_from_json method.
user_aggregates_schema = client.schema_from_json(user_aggregates_schema_path)

user_aggregates_table = bigquery.Table(project_id + '.' + dataset_id + '.user_aggregates', schema=user_aggregates_schema)
user_aggregates_table.description = "Table listing all products along with their purchasing activity. For each product, global attributes are available, like money spent in total, number of items purchased in total and last purchase date. But also daily aggregated data like number of product items or money spent for this product. This table is used for queries related to products, brands or categories."
table = client.create_table(user_aggregates_table)

# Create product_aggregates table using JSON schema
# To load a schema file use the schema_from_json method.
product_aggregates_schema = client.schema_from_json(product_aggregates_schema_path)

product_aggregates_table = bigquery.Table(project_id + '.' + dataset_id + '.product_aggregates', schema=product_aggregates_schema)
product_aggregates_table.description = "Table listing all products along with their purchasing activity. For each product, global attributes are available, like money spend spent in total, number of items purchased in total and last purchase date. But also daily aggregated data like number of product items or money spent for this product. This table is used for queries related to products, brands or categories."
table = client.create_table(product_aggregates_table)

# Reading SQL script from the file
with open(sql_file_path, "r") as sql_file:
    sql_script_tmpl = sql_file.read()
    j2_template = Template(sql_script_tmpl)
    sql_script = j2_template.render(data)

with open("generated.sql", "w") as text_file:
    text_file.write(sql_script)

query_job = client.query(sql_script, location=region)
query_job.result()  # Wait for the end of the request

print("All the SQL commands in success.")