variable "name_prefix" {
  description = "Name prefix for project resources."
  type        = string
}

variable "pipeline_role_arn" {
  description = "IAM role ARN used by the check manifest Lambda."
  type        = string
}

variable "manifest_table_name" {
  description = "DynamoDB manifest table name."
  type        = string
}

variable "lambda_runtime" {
  description = "Python runtime for Lambda."
  type        = string
}

variable "tlc_source_base_url" {
  description = "Base URL for TLC trip data parquet files."
  type        = string
}