/**
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

locals {
  image = "debian-cloud/debian-11"
  machine_type = "f1-micro"
}

resource "google_project_service" "required_api" {
  for_each = toset(["compute.googleapis.com", "cloudresourcemanager.googleapis.com"])
  service  = each.key
}

resource "google_compute_network" "failover_vpc" {
  depends_on              = [google_project_service.required_api]
  name                    = "ip-failover-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "failover_subnet" {
  name          = "ip-failover-subnet"
  ip_cidr_range = var.subnet_range
  dynamic "secondary_ip_range" {
    for_each = var.floating_ip_ranges
    content {
        range_name    = "floating-ip-${secondary_ip_range.key+1}"
        ip_cidr_range = secondary_ip_range.value
    }
  }
  network       = google_compute_network.failover_vpc.id
}

resource "google_project_iam_custom_role" "nginx_gateway_custom_role" {
  role_id     = "nginx_gateway_role"
  title       = "Nginx Gateway Custome Role"
  description = "Grant Nginx Gateways the permissions to list instances and update their network interface"
  permissions = ["compute.instances.list", "compute.instances.get", "compute.instances.updateNetworkInterface"]
}

resource "google_service_account" "nginx_gateway_sa" {
  account_id = "nginx-gateway-sa"
  display_name = "Nginx Gateway Service Account"
}

resource "google_project_iam_member" "nginx_binding" {
  project = var.project_id
  role    = "projects/${var.project_id}/roles/nginx_gateway_role"
  member  = "serviceAccount:${google_service_account.nginx_gateway_sa.email}"
}

resource "google_compute_firewall" "failover_firewall_http" {
  name = "failover-http-traffic"
  allow {
    protocol = "tcp"
    ports    = [80]
  }
  network     = google_compute_network.failover_vpc.id
  source_tags = ["client"]
  target_tags = ["nginx-gateway"]
}

resource "google_compute_firewall" "failover_firewall_ssh_iap" {
  name = "failover-ssh-iap"
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  network = google_compute_network.failover_vpc.id
  #IP range used by Identity-Aware-Proxy
  #See https://cloud.google.com/iap/docs/using-tcp-forwarding#create-firewall-rule
  source_ranges = ["35.235.240.0/20"]
}

resource "google_compute_firewall" "failover_firewall_hc" {
  name = "failover-hc"
  allow {
    protocol = "tcp"
    ports    = [var.health_check_port]
  }
  network = google_compute_network.failover_vpc.id
  #IP ranges used for health checks
  #See https://cloud.google.com/compute/docs/instance-groups/autohealing-instances-in-migs#setting_up_an_autohealing_policy
  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]
}

resource "google_compute_firewall" "failover_firewall_vrrp" {
  name = "failover-vrrp"
  allow {
    #112 is VRRP IP protocol number required for keepalived communication
    protocol = "112"
  }
  network     = google_compute_network.failover_vpc.id
  source_tags = ["nginx-gateway"]
  target_tags = ["nginx-gateway"]
}

resource "google_compute_instance_template" "nginx_primary_instance_template" {
  name_prefix  = "nginx-primary-"
  machine_type = local.machine_type
  disk {
    source_image = local.image
    auto_delete  = true
    boot         = true
  }
  metadata_startup_script = templatefile("startup-script.tmpl", {
    server_number       = 1
    health_check_port   = var.health_check_port
    floating_ip_ranges  = "(\"${join("\" \"", var.floating_ip_ranges)}\")"
    formatted_alias_ips = join(";", [for range in google_compute_subnetwork.failover_subnet.secondary_ip_range : "${range.range_name}:${range.ip_cidr_range}"])
    ip                  = var.primary_ip
    peer_ip             = var.secondary_ip
    state               = "MASTER"
    priority            = 100
    vrrp_password       = var.vrrp_password
    zone                = var.zone
  })
  tags = ["nginx-gateway"]
  
  service_account {
    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    email  = google_service_account.nginx_gateway_sa.email
    scopes = ["cloud-platform"]
  }

  network_interface {
    subnetwork = google_compute_subnetwork.failover_subnet.id
    network_ip = var.primary_ip
    access_config {}
  }

  lifecycle {
    create_before_destroy = false
  }
}
resource "google_compute_instance_template" "nginx_secondary_instance_template" {
  name_prefix  = "nginx-secondary-"
  machine_type = local.machine_type
  disk {
    source_image = local.image
    auto_delete  = true
    boot         = true
  }
  metadata_startup_script = templatefile("startup-script.tmpl", {
    server_number       = 2
    health_check_port   = var.health_check_port
    floating_ip_ranges  = "(\"${join("\" \"", var.floating_ip_ranges)}\")"
    formatted_alias_ips = join(";", [for range in google_compute_subnetwork.failover_subnet.secondary_ip_range : "${range.range_name}:${range.ip_cidr_range}"])
    ip                  = var.secondary_ip
    peer_ip             = var.primary_ip
    state               = "BACKUP"
    priority            = 100
    vrrp_password       = var.vrrp_password
    zone                = var.zone
  })
  tags = ["nginx-gateway"]
  
  service_account {
    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    email  = google_service_account.nginx_gateway_sa.email
    scopes = ["cloud-platform"]
  }
  
  network_interface {
    subnetwork = google_compute_subnetwork.failover_subnet.id
    network_ip = var.secondary_ip
    access_config {}
  }

  lifecycle {
    create_before_destroy = false
  }
}

resource "google_compute_instance_group_manager" "nginx_instance_group_primary" {
  name               = "nginx-master"
  base_instance_name = "nginx-master"
  target_size        = 1
  version {
    instance_template = google_compute_instance_template.nginx_primary_instance_template.id
  }
  auto_healing_policies {
    health_check      = google_compute_health_check.autohealing.id
    initial_delay_sec = 60
  }
}

resource "google_compute_instance_group_manager" "nginx_instance_group_secondary" {
  name               = "nginx-backup"
  base_instance_name = "nginx-backup"
  target_size        = 1
  version {
    instance_template = google_compute_instance_template.nginx_secondary_instance_template.id
  }
  auto_healing_policies {
    health_check      = google_compute_health_check.autohealing.id
    initial_delay_sec = 60
  }
}

resource "google_compute_health_check" "autohealing" {
  depends_on = [google_project_service.required_api]
  name                = "autohealing-health-check"
  check_interval_sec  = 5
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3 # 15 seconds

  http_health_check {
    request_path = "/"
    port = var.health_check_port
  }
}

resource "google_compute_instance" "client-vm" {
  name         = "client"
  machine_type = local.machine_type

  tags = ["client"]

  boot_disk {
    initialize_params {
      image = local.image
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.failover_subnet.name
  }

}