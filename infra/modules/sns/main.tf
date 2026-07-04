resource "aws_sns_topic" "pipeline_alerts" {
  name = "${var.name_prefix}-pipeline-alerts"
}

resource "aws_sns_topic_subscription" "pipeline_alert_email" {
  topic_arn = aws_sns_topic.pipeline_alerts.arn
  protocol  = "email"
  endpoint  = var.budget_alert_email
}