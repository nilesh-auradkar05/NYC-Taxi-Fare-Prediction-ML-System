output "state_machine_name" {
  description = "Step Functions state machine name."
  value       = aws_sfn_state_machine.monthly_ingestion.name
}

output "state_machine_arn" {
  description = "Step Functions state machine ARN."
  value       = aws_sfn_state_machine.monthly_ingestion.arn
}