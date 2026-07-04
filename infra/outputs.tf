output "s3_bucket_names" {
  description = "Lakehouse S3 bucket names."
  value       = module.s3_lakehouse.bucket_names
}

output "glue_database_names" {
  description = "Glue catalog databases."
  value       = module.glue_catalog.database_names
}

output "athena_workgroups" {
  description = "Athena workgroup names."
  value       = module.athena.workgroup_names
}

output "pipeline_role_arn" {
  description = "Pipeline execution role ARN."
  value       = module.iam.pipeline_role_arn
}

output "agent_readonly_role_arn" {
  description = "Agent read-only role ARN."
  value       = module.iam.agent_readonly_role_arn
}

output "pipeline_alert_topic_arn" {
  description = "SNS topic ARN for pipeline alerts."
  value       = module.sns.pipeline_alert_topic_arn
}

output "manifest_table_name" {
  description = "DynamoDB ingestion manifest table name."
  value       = module.manifest_table.table_name
}

output "fetch_lambda_function_name" {
  description = "TLC fetch Lambda function name."
  value       = module.lambda_fetch.function_name
}