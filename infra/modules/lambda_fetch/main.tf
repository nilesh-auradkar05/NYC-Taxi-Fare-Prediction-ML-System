locals {
  function_name = "${var.name_prefix}-fetch-tlc-to-bronze"
}

data "archive_file" "fetch_lambda_zip" {
  type        = "zip"
  source_dir  = "${path.root}/../ingestion/lambdas"
  output_path = "${path.root}/build/fetch_tlc_to_bronze.zip"
}

resource "aws_cloudwatch_log_group" "fetch_lambda" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 14
}

resource "aws_lambda_function" "fetch_tlc_to_bronze" {
  function_name = local.function_name
  description   = "Fetch one NYC TLC monthly parquet file to S3 bronze with manifest idempotency."

  role    = var.pipeline_role_arn
  handler = "fetch_tlc_to_bronze.handler"
  runtime = var.lambda_runtime

  filename         = data.archive_file.fetch_lambda_zip.output_path
  source_code_hash = data.archive_file.fetch_lambda_zip.output_base64sha256

  memory_size = 512
  timeout     = 900

  environment {
    variables = {
      BRONZE_BUCKET       = var.bronze_bucket_name
      MANIFEST_TABLE      = var.manifest_table_name
      TLC_SOURCE_BASE_URL = var.tlc_source_base_url
    }
  }

  depends_on = [aws_cloudwatch_log_group.fetch_lambda]
}