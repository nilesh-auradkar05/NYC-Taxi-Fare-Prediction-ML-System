output "pipeline_alert_topic_arn" {
  description = "SNS topic ARN for pipeline alerts."
  value       = aws_sns_topic.pipeline_alerts.arn
}