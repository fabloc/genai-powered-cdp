variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud Region used to deploy resources"
  default     = "europe-west1"
}

variable "artifact_registry_repo" {
  description = "Artifact Registry Repository to use to deploy the application"
  type        = string
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
}

variable "db_name" {
  description = "PgVector Database Name"
  type        = string
}

variable "db_user_name" {
  description = "PgVector User Name"
  type        = string
}

variable "db_user_password" {
  description = "PgVector User Password"
  type        = string
}

variable "provisioning_ip_address" {
  description = "Public IP address used to initialize the Cloud SQL for posgreSQL instance. It will be added as authorized IP to access the instance"
  type        = string
}