output "function_name" {
  description = "CheckManifest Lambda function name."
  value       = aws_lambda_function.check_manifest.function_name
}

output "function_arn" {
  description = "CheckManifest Lambda function ARN."
  value       = aws_lambda_function.check_manifest.arn
}