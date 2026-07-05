variable "name_prefix" {
  description = "Name prefix."
  type        = string
}

variable "aws_region" {
  description = "AWS region."
  type        = string
}

variable "account_id" {
  description = "AWS account ID."
  type        = string
}

variable "bronze_bucket_arn" {
  type = string
}

variable "silver_bucket_arn" {
  type = string
}

variable "gold_bucket_arn" {
  type = string
}

variable "artifacts_bucket_arn" {
  type = string
}

variable "bronze_bucket_name" {
  type = string
}

variable "silver_bucket_name" {
  type = string
}

variable "gold_bucket_name" {
  type = string
}

variable "artifacts_bucket_name" {
  type = string
}

variable "glue_database_names" {
  type = object({
    bronze = string
    silver = string
    gold   = string
  })
}

variable "pipeline_alert_topic_arn" {
  type = string
}

variable "manifest_table_arn" {
  description = "DynamoDB manifest table ARN."
  type        = string
}