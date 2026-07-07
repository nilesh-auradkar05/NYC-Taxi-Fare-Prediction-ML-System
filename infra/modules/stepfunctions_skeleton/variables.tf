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

variable "enable_monthly_schedule" {
  description = "Whether the monthly EventBridge Scheduler trigger is enabled."
  type        = bool
  default     = false
}

variable "monthly_schedule_expression" {
  description = "EventBridge Scheduler cron expression for monthly ingestion."
  type        = string
  default     = "cron(0 9 10 * ? *)"
}

variable "monthly_schedule_timezone" {
  description = "Timezone for the monthly EventBridge Scheduler expression."
  type        = string
  default     = "UTC"
}

variable "scheduled_service" {
  description = "TLC service sent by the monthly schedule target input."
  type        = string
  default     = "yellow"

  validation {
    condition     = contains(["yellow", "hvfhv"], var.scheduled_service)
    error_message = "scheduled_service must be one of: yellow, hvfhv."
  }
}

variable "scheduled_year_month" {
  description = "TLC year-month sent by the monthly schedule target input. Keep explicit until scheduled month inference exists."
  type        = string
  default     = "2025-01"

  validation {
    condition     = can(regex("^\\d{4}-\\d{2}$", var.scheduled_year_month))
    error_message = "scheduled_year_month must match YYYY-MM."
  }
}