variable "name_prefix" {
  description = "Name prefix."
  type        = string
}

variable "artifacts_bucket" {
  description = "Artifacts bucket used for Athena query outputs."
  type        = string
}

variable "engineering_scan_cutoff_bytes" {
  description = "Bytes scanned cutoff per query for engineering Athena workgroup."
  type        = number
}

variable "agent_scan_cutoff_bytes" {
  description = "Bytes scanned cutoff per query for agent-readonly Athena workgroup."
  type        = number
}