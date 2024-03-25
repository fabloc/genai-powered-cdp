# Demonstration of using Google Cloud to build a Customer Data Platform for user segmentation with low latency and predictive costs

## Google Disclaimer

This is not an officially supported Google Product

## Introduction

BigQuery is the platform of choice for many companies to store and analyze their data. Especially in the field of Customer Data Platform (CDP), this is a very powerful tool to batch process huge amount of data. However, one pain point with BigQuery when it comes to user facing requests: given that the model is primarily on-demand, costs can grow very fast and are not predictable. An alternative is to use capping of slots used in the BigQuery Enterprise Edition, but then performances can be severely impacted.
This demo aims at demonstrating that there are ways to mitigate both the costs predictability while still providing low latency.

On top of that, this demonstration aims at showing that Generative AI is also a viable way to address another (more general) pain point with CDP: user exploration when defining marketing segments of users.
Most of the times, CDPs offer multiple ways to explore and define the user segments. It ranges from the pre-arranged filters to a custom query language. The issue comes from the fact that the end users of a CDP are marketing personas, who have little to no technical knowledge, and exploring the datasets can become tedious for them, and source of ongoing support from the CDP vendor.
This project provides a way to query multiple BigQuery datasets and get an answers under 10s, with the ability to understand the intent of the user and select the appropriate tables to query. It then translates the user question into a BigQuery SQL query and executes it.
There are several different ways to do nl2sql, but in the case of a CDP, where the end user will not have access to BigQuery, it is paramount that the generated SQL query is syntactically correct and actually answers the user's question. These 2 aspects are what is lacking in the common nl2sql libraries, and what this project also addresses.

## Architecture

To achieve this, the demo uses the following technologies:
- Cloud Run
- Cloud Build
- Artifact Registry
- Vertex AI
- BigQuery
- Cloud SQL for PosgreSQL with pgvector extension
- Streamlit Framework
- Python 3.10


## Repository structure

```
.
├── app
└── config
    └── queries_samples
└── installation_scripts
    └── bigquery_dataset
    └── terraform
```

- [`/app`](/app): Source code for demo app.  
- [`/config`](/config): Configuration files used by the application.
- [`/config/queries_samples`](/config/queries_samples): Sample questions with associated SQL queries that will be ingested inside the Vector Database during the provisioning of the resources.
- [`/installation_scripts`](/installation_scripts): Scripts used to install the application.
- [`/bigquery_dataset`](/installation_scripts/bigquery_dataset): Scripts used to import and generate the tables that will be used for the CDP demo.
- [`/installation_scripts/terraform`](/installation_scripts/terraform): Terraform files .


# Environment Setup

Please follow the instructions detailed in the page [`/installation_scripts`](/installation_scripts) to set up the environment.


## Getting help

If you have any questions or if you found any problems with this repository, please report through GitHub issues.
