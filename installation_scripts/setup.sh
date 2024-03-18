#!/bin/bash
#
# Performs an automated installation of the Customer Data Platform
# Modify the Globals variables prior to running this script
#################################

# Global variables
#################################
PROJECT_ID="genai-cdp-test-3"                   # ID of the project where you want to deploy
REGION="europe-west1"                           # Name of the region
AUTH_USER="admin@fabienlocquet.altostrat.com"   # User that will run the application
ARTIFACT_REGISTRY_REPO="genai-cdp-repo"         # Name of the Artifact Registry Repository
SERVICE_NAME="genai-cdp"                        # Name of the Cloud Run Service
DATABASE_NAME="nl2sql-rag-db"
DATABASE_USER="nl2sql-admin"
DATABASE_PASSWORD=">rJFj8HbN<:ObiEm"
BIGQUERY_DATASET="cdp_demo"
EVENTS_START_DATE="2020-01-01"
#################################


# do not modify below here

function check_if_project_id_is_setup() {
    if [ -z "$PROJECT_ID" ]; then
        echo "Error: You must configure your PROJECT_ID."
        exit 1
    fi
}


function check_gcloud_authentication() {
    # Check if the user is authenticated with gcloud
    local AUTHENTICATED_USER=$(gcloud auth list --format="value(account)" --filter="status:ACTIVE")

    if [ -z "$AUTHENTICATED_USER" ]; then
    echo "No authenticated user found. Please authenticate using 'gcloud auth login'."
    exit 1
    else
    echo "Authenticated user is: $AUTHENTICATED_USER"
    fi
}

function check_gcp_project() {
# Check if the project exists
local PROJECT_NAME=$(gcloud projects describe ${PROJECT_ID//\"} --format="value(name)")

 if [ -z "$PROJECT_NAME" ]; then
   echo "Project $PROJECT_ID does not exist."
   exit 1
 else
   echo "Project $PROJECT_ID exists."
 fi

 # Check if the environment is configured for the project
 local CONFIGURED_PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

 if [ "$CONFIGURED_PROJECT_ID" != "$PROJECT_ID" ]; then
   echo "Current environment is not configured for project $PROJECT_ID. Please run 'gcloud config set project $PROJECT_ID'."
   exit 1
 else
   echo "Environment is configured for project $PROJECT_ID."
 fi
}


function check_gcp_constraints() {
 local CONSTRAINTS=(
   "iam.allowedPolicyMemberDomains"
 )


 for CONSTRAINT in "${CONSTRAINTS[@]}"
 do
   local CONSTRAINT_STATUS=$(gcloud alpha resource-manager org-policies describe --effective --project=$PROJECT_ID $CONSTRAINT | sed 's/booleanPolicy: {}/ALLOW/' | grep -E 'constraint:|ALLOW' | awk '/ALLOW/ {print "allowed"}')


   if [ -z "$CONSTRAINT_STATUS" ]; then
     echo "Constraint $CONSTRAINT not found or not configured for this project."
     echo "Please ensure that the $CONSTRAINT constraint is authorized."
     exit 1
   elif [ "$CONSTRAINT_STATUS" = "allowed" ]; then
     echo "Constraint $CONSTRAINT is allowed."
   else
     echo "Constraint $CONSTRAINT is not allowed."
     echo "Please ensure that the $CONSTRAINT constraint is authorized."
     exit 1
   fi
 done
}

# Running checks before deploy
echo ""
echo "Running pre-checks"
echo ""
check_if_project_id_is_setup

# Check authentication
echo "***** Checking authentication with gcloud *****"
check_gcloud_authentication

# Check project configuration
echo "***** Checking project configuration *****"
check_gcp_project

# Check project constraints
echo "***** Checking project constraints *****"
check_gcp_constraints


# Enabling the services
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com aiplatform.googleapis.com compute.googleapis.com bigquery.googleapis.com
gcloud services enable servicenetworking.googleapis.com cloudresourcemanager.googleapis.com sqladmin.googleapis.com

cp -f variables.auto.tfvars.tmpl terraform/variables.auto.tfvars
cp -f config.ini.tmpl ../config/config.ini
cp -f bigquery_dataset/config.ini.tmpl bigquery_dataset/config.ini

# Updating the Project and Location details in app config and override files
sed -i "s|project_id = \"\"|project_id = \"${PROJECT_ID}\"|" terraform/variables.auto.tfvars
sed -i "s|region = \"\"|region = \"${REGION}\"|" terraform/variables.auto.tfvars
sed -i "s|artifact_registry_repo = \"\"|artifact_registry_repo = \"${ARTIFACT_REGISTRY_REPO}\"|" terraform/variables.auto.tfvars
sed -i "s|service_name = \"\"|service_name = \"${SERVICE_NAME}\"|" terraform/variables.auto.tfvars
sed -i "s|db_name = \"\"|db_name = \"${DATABASE_NAME}\"|" terraform/variables.auto.tfvars
sed -i "s|db_user_name = \"\"|db_user_name = \"${DATABASE_USER}\"|" terraform/variables.auto.tfvars
sed -i "s|db_user_password = \"\"|db_user_password = \"${DATABASE_PASSWORD}\"|" terraform/variables.auto.tfvars
sed -i "s|provisioning_ip_address = \"\"|provisioning_ip_address = \"$(wget http://ipinfo.io/ip -qO - --timeout=20)\"|" terraform/variables.auto.tfvars
sed -i "s|project_id =|project_id = ${PROJECT_ID}|" ../config/config.ini
sed -i "s|region =|region = ${REGION}|" ../config/config.ini
sed -i "s|auth_user =|auth_user = ${AUTH_USER}|" ../config/config.ini
sed -i "s|project_id_data =|project_id_data = ${PROJECT_ID}|" ../config/config.ini
sed -i "s|dataset_id =|dataset_id = ${BIGQUERY_DATASET}|" ../config/config.ini
sed -i "s|database_name =|database_name = ${DATABASE_NAME}|" ../config/config.ini
sed -i "s|database_user =|database_user = ${DATABASE_USER}|" ../config/config.ini
sed -i "s|database_password =|database_password = ${DATABASE_PASSWORD}|" ../config/config.ini
sed -i "s|dataset_id =|dataset_id = ${BIGQUERY_DATASET}|" bigquery_dataset/config.ini
sed -i "s|region =|region = ${REGION}|" bigquery_dataset/config.ini
sed -i "s|project_id =|project_id = ${PROJECT_ID}|" bigquery_dataset/config.ini
sed -i "s|events_start_date =|events_start_date = ${EVENTS_START_DATE}|" bigquery_dataset/config.ini

# Starting Configuration
echo "***** Create a new Artifact Repository for our webapp *****"
gcloud artifacts repositories create "$ARTIFACT_REGISTRY_REPO" --location="$REGION" --repository-format=Docker > /dev/null
echo "***** Repository created *****"

echo "***** Setup artefact docker authentication *****"
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet > /dev/null

echo "***** Build WebApp Docker image *****"
cd ..
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/$SERVICE_NAME" > /dev/null

echo "***** Checking Terraform Installation *****"
if ! command -v terraform version &> /dev/null
then
    echo "Terraform is not installed, please install it and try again."
    exit 1
else
    echo "Terraform executable found"
fi

echo "***** Initialize Terraform *****"
cd installation_scripts/terraform
terraform init

echo "***** Deploying Infrastructure using Terraform *****"
terraform apply -auto-approve

echo "***** Cloud RUN URL *****"
APP_URL=$(gcloud run services describe $SERVICE_NAME --region="$REGION" --format="value(status.url)")
echo $APP_URL

cd ../../app

if [ ! -d ".venv" ]; then   # Checking the Virtualenv folder exists or not
   python3 -m venv .venv    # Creating virtualenv  
fi

source .venv/bin/activate   # activate Virtualenv

# installing required python packages
pip install -r requirements.txt

# echo "***** Initializing Bigquerywith new Dataset and all required Tables *****"
cd ..
python3 installation_scripts/bigquery_dataset/init_bigquery.py

echo "***** Initializing PgVector Database with tables definitions and samples *****"
sed -i "s|sql_ip_type = PRIVATE|sql_ip_type = PUBLIC|" config/config.ini
python3 app/init_pgvector.py

echo "***** Removing public IP from Cloud SQL instance *****"
gcloud sql instances patch pgvector-db --no-assign-ip