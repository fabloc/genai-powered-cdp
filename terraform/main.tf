provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
resource "google_compute_network" "cdp_vpc" {
  name = "cdp-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "cdp_subnet" {
  name                      = "cdp-subnetwork"
  ip_cidr_range             = "10.0.0.0/24"
  region                    = "europe-west1"
  network                   = google_compute_network.cdp_vpc.id
  private_ip_google_access  = true
}

resource "google_compute_global_address" "private_ip_address" {
  provider = google-beta

  name          = "private-ip-address"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.cdp_vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  provider = google-beta

  network                 = google_compute_network.cdp_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Cloud SQL for PostgreSQL (Private IP)
resource "google_sql_database_instance" "pgvector_db" {
  provider = google-beta

  name             = "pgvector-db"
  project = var.project_id
  database_version = "POSTGRES_15"
  region           = var.region

  deletion_protection = false

  depends_on = [google_service_networking_connection.private_vpc_connection]

  settings {
    tier = "db-f1-micro"
    ip_configuration {
      ipv4_enabled    = false // Enforce private IP only
      private_network = google_compute_network.cdp_vpc.id
      enable_private_path_for_google_cloud_services = true
    }
  }
}

resource "google_sql_database" "database" {
  name     = var.db_name
  instance = google_sql_database_instance.pgvector_db.name
}

resource "google_sql_user" "nl2sql_user" {
  name     = var.db_user_name
  instance = google_sql_database_instance.pgvector_db.name
  password = var.db_user_password
}

resource "google_service_account" "cloudrun_sa" {
  account_id   = "${var.project_id}-cloudrun-sa"
  display_name = "Nl2SQL Cloud Run Service Account for accessing Cloud SQL"
}

resource "google_project_iam_member" "cloudrun_sa_bigquery_user" {
  project = var.project_id
  role    = "roles/bigquery.user"
  member  = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

resource "google_project_iam_member" "cloudrun_sa_bigquery_dataviewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

resource "google_project_iam_member" "cloudrun_sa_db_access" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

resource "google_project_iam_member" "cloudrun_sa_vertexai" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cloudrun_sa.email}"
}

# Cloud Run service 
resource "google_cloud_run_v2_service" "cloud_run_service" {
  provider = google-beta
  name     = var.service_name
  project = var.project_id
  location = var.region
  launch_stage = "BETA"

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo}/${var.service_name}"
      ports {
        container_port = 8501
      }
    }

    vpc_access{
      network_interfaces {
        network = google_compute_network.cdp_vpc.id
        subnetwork = google_compute_subnetwork.cdp_subnet.id
      }
      egress = "ALL_TRAFFIC"
    }

    service_account = google_service_account.cloudrun_sa.email
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
  }
}

resource "google_cloud_run_service_iam_binding" "default" {
  location = google_cloud_run_v2_service.cloud_run_service.location
  service  = google_cloud_run_v2_service.cloud_run_service.name
  role     = "roles/run.invoker"
  members = [
    "allUsers"
  ]
}