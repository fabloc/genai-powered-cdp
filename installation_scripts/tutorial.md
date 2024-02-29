# Setting up Generative AI for Customer Data Platform Solution

## Welcome!
Follow this instructions to deploy the Generative AI Customer Data Platform Solution within your Project.

To see more details about the solution, components and architecture, please go to the main [README](https://github.com/fabloc/genai-powered-cdp/blob/main/README.md) file.

**Time to complete**: About **TBD** minutes

Click the **Start** button to move to the next step.

## Prerequisites
Before start please make sure you have the following prerequisites:
- A GCP Project, this can be a standalone project or within a GCP organization. If you need to create a new project please follow the instructions [here](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
- A user with permissions to execute the installation script with the following permissions at the project project level:
  - Project IAM Admin
  - Service Usage Admin
  - Service Account Admin
  - BigQuery Admin
  - Cloud SQL Admin
  - Artifact Registry Admin
  - Cloud Build Admin
  - Cloud Run Admin
  - Compute Engine Admin

You can assign missing roles or view your current permissions [here](https://console.cloud.google.com/iam-admin/iam)

(Optional) If you are not executing this using cloud shell you will need to install the following:
- python 3.10 or higher
- virtualenv
- [gcloud](https://cloud.google.com/sdk/docs/install)

Make sure you set working project Id by executing: 
```bash
gcloud config set project <project_id>
```
Continue on to the next step to start the deployment process.

## Deployment

### (Optional) Advanced configuration
You can overwrite default parameters as regions or some specific values by editing the <walkthrough-editor-open-file
    filePath="genai-powered-cdp/installation_scripts/setup.sh">
    setup.sh
</walkthrough-editor-open-file> script.

### Start the deployment
Start the deployment using the following command:
```bash
cd installation_scripts && sh setup.sh
```

The script performs the following steps:
1. Check if the value of PROJECT_ID is set on the file setup.sh
2. Check if the Cloud Shell user is authenticated
3. Check if the PROJECT_ID exists
4. Check if the current environment of Cloud Shell is configured for the project
5. Check if the GCP project is not enforced with the constraint "iam.allowedPolicyMemberDomains"
6. Change the value of PROJECT_ID, LOCATION into the main.py python file according to the value of the script setup.sh
7. Enable Artifact Registry, Cloud Build, Cloud Run, Vertex AI, Bigquery, Cloud SQL and Compute Engine APIs
8. Creates the BigQuery Dataset and Tables
9. Populate the BigQuery Tables with generated data
10. Create a new Artifact Repository for the App
11. Setup artefact Docker Authentication
12. Build the Docker image of the App with Cloud Build
13. Execute a terraform script that creates the required VPC and subnet, deploys the Cloud SQL instance with Private IP and create the databases, deploys the Cloud Run service with VPC Direct Access and configure the Cloud Run service account so that it can access the Cloud SQL instance and BigQuery
14. Show the Cloud Run URL of the App


For a more detailed information please refer to the main [README](https://github.com/fabloc/genai-powered-cdp/blob/main/README.md) file.