#!/bin/bash

# Reads the value of a property from a properties file.
#
# $1 - Key name, matched at beginning of line.
function prop {
    grep "^${1}" terraform/variables.auto.tfvars | cut -d'=' -f2 | cut -d "\"" -f 2
}

# Set the project ID
PROJECT_ID=$(prop "project_id")
REGION=$(prop "region")
ARTIFACT_REGISTRY_REPO=$(prop 'artifact_registry_repo')
SERVICE_NAME=$(prop 'service_name')

echo "***** Project ID: $PROJECT_ID *****"
echo "***** Region: $REGION *****"
echo "***** Artifact Registry Repository: $ARTIFACT_REGISTRY_REPO *****"
echo "***** Service Name: $SERVICE_NAME *****"


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

# Enable APIs
echo "***** Enabling Artifact Registry, Cloud Build, Cloud Run, Vertex AI, BigQuery, Compute APIs and Networking Services *****"
gcloud services enable artifactregistry.googleapis.com > /dev/null
gcloud services enable cloudbuild.googleapis.com > /dev/null
gcloud services enable run.googleapis.com > /dev/null
gcloud services enable aiplatform.googleapis.com > /dev/null
gcloud services enable compute.googleapis.com > /dev/null
gcloud services enable bigquery.googleapis.com > /dev/null
gcloud services enable servicenetworking.googleapis.com > /dev/null

echo "***** APIs enabled *****"

# Starting Configuration
echo "***** Create a new Artifact Repository for our webapp *****"
gcloud artifacts repositories create "$ARTIFACT_REGISTRY_REPO" --location="$REGION" --repository-format=Docker > /dev/null
echo "***** Repository created *****"

echo "***** Setup artefact docker authentication *****"
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet > /dev/null

echo "***** Build WebApp Docker image *****"
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
cd terraform
terraform init

echo "***** Deploying Infrastructure using Terraform *****"
terraform apply -auto-approve

echo "***** Cloud RUN URL *****"
APP_URL=$(gcloud run services describe $SERVICE_NAME --region="$REGION" --format="value(status.url)")
echo $APP_URL