variable "name_prefix" {
  description = "Name prefix for project resources."
  type        = string
}

variable "account_id" {
  description = "AWS account ID."
  type        = string
}

variable "check_manifest_lambda_arn" {
  description = "ARN of the CheckManifest Lambda."
  type        = string
}

variable "fetch_lambda_arn" {
  description = "ARN of the FetchToBronze Lambda."
  type        = string
}

variable "alert_topic_arn" {
  description = "SNS topic ARN for Step Functions failure alerts."
  type        = string
}

variable "conform_job_name" {
  description = "AWS Glue conform job name."
  type        = string
}

variable "dq_gate_job_name" {
  description = "AWS Glue DQ gate job name."
  type        = string
}