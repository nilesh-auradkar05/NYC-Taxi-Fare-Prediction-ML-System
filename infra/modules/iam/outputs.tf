output "pipeline_role_arn" {
  description = "Pipeline execution role ARN."
  value       = aws_iam_role.pipeline_execution.arn
}

output "agent_readonly_role_arn" {
  description = "Agent read-only role ARN."
  value       = aws_iam_role.agent_readonly.arn
}