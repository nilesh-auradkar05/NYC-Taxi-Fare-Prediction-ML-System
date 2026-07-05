locals {
  glue_conform_script_key = "glue/jobs/conform.py"
}

resource "aws_s3_object" "glue_conform_script" {
  bucket = var.artifacts_bucket_name
  key    = local.glue_conform_script_key
  source = "${path.module}/../ingestion/jobs/conform.py"
  etag   = filemd5("${path.module}/../ingestion/jobs/conform.py")
}

resource "aws_glue_job" "conform_trips" {
  name              = "${var.project_name}-conform-trips"
  role_arn          = module.iam.pipeline_role_arn
  glue_version      = var.glue_version
  worker_type       = "G.1X"
  number_of_workers = 2
  timeout           = 30
  max_retries       = 1

  command {
    name            = "glueetl"
    script_location = "s3://${var.artifacts_bucket_name}/${aws_s3_object.glue_conform_script.key}"
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"                     = "python"
    "--enable-glue-datacatalog"          = "true"
    "--datalake-formats"                 = "iceberg"
    "--enable-metrics"                   = "true"
    "--enable-continuous-cloudwatch-log" = "true"

    "--spark-event-logs-path" = "s3://${var.artifacts_bucket_name}/glue/spark-event-logs/"

    "--conf" = join(" ", [
      "spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
      "--conf spark.sql.catalog.glue_catalog=org.apache.iceberg.spark.SparkCatalog",
      "--conf spark.sql.catalog.glue_catalog.warehouse=s3://${var.silver_bucket_name}/iceberg/",
      "--conf spark.sql.catalog.glue_catalog.catalog-impl=org.apache.iceberg.aws.glue.GlueCatalog",
      "--conf spark.sql.catalog.glue_catalog.io-impl=org.apache.iceberg.aws.s3.S3FileIO"
    ])

    "--catalog"          = "glue_catalog"
    "--silver_db"        = var.silver_glue_database_name
    "--bronze_base_path" = "s3://${var.bronze_bucket_name}/bronze"
    "--zone_lookup_path" = "s3://${var.artifacts_bucket_name}/reference/taxi_zone_lookup.csv"
    "--temp_base_path"   = "s3://${var.artifacts_bucket_name}/tmp/glue/conform-merge-staging"
  }

  execution_property {
    max_concurrent_runs = 1
  }
}