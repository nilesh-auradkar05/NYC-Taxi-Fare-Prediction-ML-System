locals {
  state_machine_name = "${var.name_prefix}-monthly-ingestion"
}

data "aws_iam_policy_document" "assume_stepfunctions" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "stepfunctions_role" {
  name               = "${var.name_prefix}-stepfunctions-role"
  assume_role_policy = data.aws_iam_policy_document.assume_stepfunctions.json
}

resource "aws_iam_role_policy" "sfn_glue_dq_sns" {
  name = "${var.name_prefix}-sfn-glue-dq-sns"
  role = aws_iam_role.stepfunctions_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "glue:StartJobRun",
          "glue:GetJobRun",
          "glue:GetJobRuns",
          "glue:BatchStopJobRun"
        ]
        Resource = ["arn:aws:glue:us-east-1:${var.account_id}:job/nyc-taxi-prediction-dataops-dev-*"]
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = var.alert_topic_arn
      }
    ]
  })
}

data "aws_iam_policy_document" "stepfunctions_policy" {
  statement {
    sid    = "InvokePipelineLambdas"
    effect = "Allow"

    actions = [
      "lambda:InvokeFunction"
    ]

    resources = [
      var.check_manifest_lambda_arn,
      "${var.check_manifest_lambda_arn}:*",
      var.fetch_lambda_arn,
      "${var.fetch_lambda_arn}:*"
    ]
  }
}

resource "aws_iam_policy" "stepfunctions_policy" {
  name        = "${var.name_prefix}-stepfunctions-policy"
  description = "Allow Step Functions to invoke ingestion Lambdas."
  policy      = data.aws_iam_policy_document.stepfunctions_policy.json
}

resource "aws_iam_role_policy_attachment" "stepfunctions_policy" {
  role       = aws_iam_role.stepfunctions_role.name
  policy_arn = aws_iam_policy.stepfunctions_policy.arn
}

resource "aws_sfn_state_machine" "monthly_ingestion" {
  name     = local.state_machine_name
  role_arn = aws_iam_role.stepfunctions_role.arn
  type     = "STANDARD"

  definition = templatefile("${path.module}/step_functions/ingestion.asl.json.tftpl", {
    check_manifest_lambda_arn  = var.check_manifest_lambda_arn
    fetch_to_bronze_lambda_arn = var.fetch_lambda_arn
    conform_job_name           = var.conform_job_name
    dq_gate_job_name           = var.dq_gate_job_name
    alert_topic_arn            = var.alert_topic_arn
  })
}