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

  definition = jsonencode({
    Comment = "T-006 skeleton: CheckManifest -> Choice(new?) -> FetchToBronze -> Succeed"
    StartAt = "CheckManifest"

    States = {
      CheckManifest = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = var.check_manifest_lambda_arn
          "Payload.$"  = "$"
        }
        OutputPath = "$.Payload"
        Retry = [
          {
            ErrorEquals = [
              "Lambda.ServiceException",
              "Lambda.AWSLambdaException",
              "Lambda.SdkClientException",
              "States.TaskFailed"
            ]
            IntervalSeconds = 2
            MaxAttempts     = 3
            BackoffRate     = 2.0
          }
        ]
        Next = "IsNewFile"
      }

      IsNewFile = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.is_new"
            BooleanEquals = true
            Next          = "FetchToBronze"
          }
        ]
        Default = "AlreadyFetched"
      }

      AlreadyFetched = {
        Type = "Succeed"
      }

      FetchToBronze = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = var.fetch_lambda_arn
          "Payload.$"  = "$"
        }
        OutputPath = "$.Payload"
        Retry = [
          {
            ErrorEquals = [
              "Lambda.ServiceException",
              "Lambda.AWSLambdaException",
              "Lambda.SdkClientException",
              "States.TaskFailed"
            ]
            IntervalSeconds = 2
            MaxAttempts     = 3
            BackoffRate     = 2.0
          }
        ]
        End = true
      }
    }
  })
}