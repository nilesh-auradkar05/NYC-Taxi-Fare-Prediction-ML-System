variable "project_name" {
  description = "Project slug used for AWS resource names."
  type        = string
  default     = "nyc-mobility-v3"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for project resources."
  type        = string
  default     = "us-east-1"
}

variable "budget_alert_email" {
  description = "Email address for AWS Budget alerts. Required for T-004."
  type        = string
}

variable "force_destroy_buckets" {
  description = "Whether Terraform can destroy non-empty S3 buckets. Keep false unless tearing down deliberately."
  type        = bool
  default     = false
}

variable "engineering_athena_scan_cutoff_bytes" {
  description = "Athena scanned-byte cutoff per query for engineering workgroup."
  type        = number
  default     = 1073741824 # 1 GB
}

variable "agent_athena_scan_cutoff_bytes" {
  description = "Athena scanned-byte cutoff per query for agent-readonly workgroup."
  type        = number
  default     = 104857600 # 100 MB
}