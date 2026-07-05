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

variable "lambda_runtime" {
  description = "Python runtime for Lambda functions."
  type        = string
  default     = "python3.14"
}

variable "tlc_source_base_url" {
  description = "Base URL for TLC trip data parquet files."
  type        = string
  default     = "https://d37ci6vzurychx.cloudfront.net/trip-data"
}

variable "artifacts_bucket_name" {
  description = "Existing T-004 artifacts bucket name used for Glue scripts and reference files."
  type        = string
}

variable "bronze_bucket_name" {
  description = "Existing T-004 bronze lakehouse bucket name."
  type        = string
}

variable "silver_bucket_name" {
  description = "Existing T-004 silver lakehouse bucket name."
  type        = string
}

variable "silver_glue_database_name" {
  description = "Existing T-004 Glue Catalog database name for silver tables."
  type        = string
}

variable "glue_version" {
  description = "AWS Glue version for the conform job."
  type        = string
  default     = "5.1"
}