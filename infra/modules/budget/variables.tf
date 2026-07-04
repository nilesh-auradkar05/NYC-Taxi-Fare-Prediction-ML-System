variable "name_prefix" {
  description = "Name prefix."
  type        = string
}

variable "budget_alert_email" {
  description = "Email address for AWS Budget notifications."
  type        = string
}

variable "monthly_limit_usd" {
  description = "Monthly cost budget limit in USD."
  type        = string
}