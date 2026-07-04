locals {
  workgroups = {
    engineering = {
      name          = "engineering"
      scan_cutoff   = var.engineering_scan_cutoff_bytes
      output_prefix = "athena-results/engineering/"
    }

    agent_readonly = {
      name          = "agent-readonly"
      scan_cutoff   = var.agent_scan_cutoff_bytes
      output_prefix = "athena-results/agent-readonly/"
    }
  }
}

resource "aws_athena_workgroup" "this" {
  for_each = local.workgroups

  name        = each.value.name
  description = "NYC Mobility ${each.value.name} Athena workgroup."
  state       = "ENABLED"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true
    bytes_scanned_cutoff_per_query     = each.value.scan_cutoff

    result_configuration {
      output_location = "s3://${var.artifacts_bucket}/${each.value.output_prefix}"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
}