variable "name_prefix" {
  description = "Prefix for bucket names."
  type        = string
}

variable "account_id" {
  description = "AWS account ID used to make bucket names globally unique."
  type        = string
}

variable "force_destroy" {
  description = "Whether buckets can be destroyed even when non-empty."
  type        = bool
  default     = false
}