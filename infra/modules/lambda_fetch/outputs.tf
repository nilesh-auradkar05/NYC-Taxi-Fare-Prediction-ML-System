output "function_name" {
  description = "Fetch Lambda function name."
  value       = aws_lambda_function.fetch_tlc_to_bronze.function_name
}

output "function_arn" {
  description = "Fetch Lambda function ARN."
  value       = aws_lambda_function.fetch_tlc_to_bronze.arn
}