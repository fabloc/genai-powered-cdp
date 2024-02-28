from google.cloud import bigquery
from google.api_core.exceptions import Conflict
import subprocess

# Default values
project_id = "Your project id"  # Change the value here
dataset_id = "demo"
sql_file_path = "bigquery_demo_dataset.sql"

try:
        subprocess.run(["gcloud", "services", "enable", "bigquery.googleapis.com", "--project", project_id], check=True)
        print("BigQuery API has been enabled.")

except subprocess.CalledProcessError:
    print("Failed to enable BigQuery API. Please check your gcloud configuration and permissions.")

# Enable BigQuery Client
client = bigquery.Client(project=project_id)

# Setup of the Dataset
dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
dataset = bigquery.Dataset(dataset_ref)

try:
    client.create_dataset(dataset)
    print(f"Dataset {dataset_id} created with success.")
except Conflict:
    print(f"Dataset {dataset_id} already exists.")

# Reading SQL script from the file
with open(sql_file_path, "r") as sql_file:
    sql_script = sql_file.read()

# Split the script in multiples commands if needed
sql_commands = sql_script.split(';')[:-1]

# Run every SQL commands in the file
for command in sql_commands:
    if command.strip():  # Check if the command is not empty
        query_job = client.query(command, location="US")
        query_job.result()  # Wait for the end of the request
        print("SQL Command in success.")

print("All the SQL commands in success.")
