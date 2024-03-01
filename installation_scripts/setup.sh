#!/bin/bash
#
# Performs an automated installation of the Customer Data Platform
# Modify the Globals variables prior to running this script
#################################

# Global variables
#################################
PROJECT_ID="genai-cdp-test"                  # ID of the project where you want to deploy
REGION="europe-west1"                     # Name of the region
DATASET_NAME="cdp-dataset"                # BigQuery Dataset Name for creation
ARTIFACT_REGISTRY_REPO="genai-cdp-repo"   # Name of the Artifact Registry Repository
SERVICE_NAME="genai-cdp"                  # Name of the Cloud Run Service
DATABASE_NAME="nl2sql-rag-db"
DATABASE_USER="nl2sql-admin"
DATABASE_PASSWORD=">rJFj8HbN<:ObiEm"
#################################


# do not modify below here


# read -p "Enter Project ID: " PROJECT_ID
# read -p "Enter region: " REGION

echo -e "\n \n"
bold=$(tput bold)
normal=$(tput sgr0)
echo -e "Here are the names that will be used for creating resources \n"
echo -e "BIGQUERY DATASET_NAME: ${bold}${DATASET_NAME}${normal} \ARTIFACT_REGISTRY_REPO: ${bold}${ARTIFACT_REGISTRY_REPO}${normal} \SERVICE_NAME: ${bold}${SERVICE_NAME}${normal}"
echo -e "\nDo you wish to add postfix ? enter 1 for Yes and 2 for No"
echo -e "\n Note: If you are reruning the automation, use names like earlier"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) read -p "Enter Postfix: " POSTFIX
              DATASET_NAME="${DATASET_NAME}_${POSTFIX}"
              ARTIFACT_REGISTRY_REPO="${ARTIFACT_REGISTRY_REPO}_${POSTFIX}"
              SERVICE_NAME="${SERVICE_NAME}_${POSTFIX}"
              break ;;
        No ) echo "Using same names for setup"
             break ;;
    esac
done

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

# if [ ! -d "genai_powered_cdp" ]; then   # Checking the Virtualenv folder exists or not
#   python3 -m venv genai_powered_cdp    # Creating virtualenv  
# fi

# source genai_powered_cdp/bin/activate   # activate Virtualenv

# installing required python packages
# pip install -U google-cloud-datacatalog google-cloud-storage google-cloud-bigquery numpy google-api-python-client google.cloud google.auth google-cloud-discoveryengine google-cloud-dialogflow-cx

#-----BigQuery Setup -----
# python3 genai_marketing_env_setup.py $PROJECT_ID $REGION $DATASET_NAME

if [ ! -d "genai-powered-cdp" ]; then   # Checking the github code folder exists or not
  git clone https://github.com/fabloc/genai-powered-cdp   # cloning the genai-for-marketing code from github
else
  rm -rf genai-for-marketing
  git clone https://github.com/fabloc/genai-powered-cdp

fi

# Updating the Project and Location details in app config and override files
sed -i "s|project_id = \"\"|project_id = \"${PROJECT_ID}\"|" terraform/variables.auto.tfvars
sed -i "s|region = \"\"|region = \"${REGION}\"|" terraform/variables.auto.tfvars
sed -i "s|artifact_registry_repo = \"\"|artifact_registry_repo = \"${ARTIFACT_REGISTRY_REPO}\"|" terraform/variables.auto.tfvars
sed -i "s|service_name = \"\"|service_name = \"${SERVICE_NAME}\"|" terraform/variables.auto.tfvars
sed -i "s|database_name = \"\"|database_name = \"${DATABASE_NAME}\"|" terraform/variables.auto.tfvars
sed -i "s|database_user = \"\"|database_user = \"${DATABASE_USER}\"|" terraform/variables.auto.tfvars
sed -i "s|database_password = \"\"|database_password = \"${DATABASE_PASSWORD}\"|" terraform/variables.auto.tfvars
sed -i "s|project_id =|project_id = ${PROJECT_ID}|" config/config.ini
sed -i "s|region = |region = ${REGION}|" config/config.ini
sed -i "s|database_name =|database_name = ${DATABASE_NAME}|" config/config.ini
sed -i "s|database_user = |database_user = ${DATABASE_USER}|" config/config.ini
sed -i "s|database_password = |database_password = ${DATABASE_PASSWORD}|" config/config.ini

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