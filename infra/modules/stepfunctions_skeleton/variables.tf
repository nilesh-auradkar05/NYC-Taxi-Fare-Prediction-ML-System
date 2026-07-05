variable "name_prefix" {
  description = "Name prefix for project resources."
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