locals {
  function_name = "${var.name_prefix}-check-manifest"
}

data "archive_file" "check_manifest_zip" {
  type        = "zip"
  source_dir  = "${path.root}/../ingestion/lambdas"
  output_path = "${path.root}/build/check_manifest.zip"
}

resource "aws_cloudwatch_log_group" "check_manifest" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 14
}

resource "aws_lambda_function" "check_manifest" {
  function_name = local.function_name
  description   = "Check TLC source metadata and DynamoDB manifest idempotency."

  role    = var.pipeline_role_arn
  handler = "check_manifest.handler"
  runtime = var.lambda_runtime

  filename         = data.archive_file.check_manifest_zip.output_path
  source_code_hash = data.archive_file.check_manifest_zip.output_base64sha256

  memory_size = 256
  timeout     = 60

  environment {
    variables = {
      MANIFEST_TABLE      = var.manifest_table_name
      TLC_SOURCE_BASE_URL = var.tlc_source_base_url
    }
  }

  depends_on = [aws_cloudwatch_log_group.check_manifest]
}