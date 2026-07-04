variable "name_prefix" {
  description = "Name prefix for project resources."
  type        = string
}

variable "pipeline_role_arn" {
  description = "IAM role ARN used by the fetch Lambda."
  type        = string
}

variable "bronze_bucket_name" {
  description = "Bronze S3 bucket name."
  type        = string
}

variable "manifest_table_name" {
  description = "DynamoDB manifest table name."
  type        = string
}

variable "lambda_runtime" {
  description = "Python runtime for Lambda."
  type        = string
  default     = "python3.14"
}

variable "tlc_source_base_url" {
  description = "Base URL for TLC trip data parquet files."
  type        = string
  default     = "https://d37ci6vzurychx.cloudfront.net/trip-data"
}