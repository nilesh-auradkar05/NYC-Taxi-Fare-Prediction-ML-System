output "table_name" {
  description = "DynamoDB manifest table name."
  value       = aws_dynamodb_table.manifest.name
}

output "table_arn" {
  description = "DynamoDB manifest table ARN."
  value       = aws_dynamodb_table.manifest.arn
}