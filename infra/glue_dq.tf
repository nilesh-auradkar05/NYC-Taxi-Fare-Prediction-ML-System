locals {
  dq_rules_key       = "glue/dq/silver_trips_dq.dqdl"
  dq_config_key      = "glue/dq/silver_trips_dq_config.json"
  dq_gate_script_key = "glue/jobs/dq_gate.py"
  crz_zones_seed_key = "reference/crz_zones.csv"
}

resource "aws_s3_object" "dq_rules_silver_trips" {
  bucket = var.artifacts_bucket_name
  key    = local.dq_rules_key
  source = "${path.module}/../ingestion/dq/rules/silver_trips_dq.dqdl"
  etag   = filemd5("${path.module}/../ingestion/dq/rules/silver_trips_dq.dqdl")
}

resource "aws_s3_object" "dq_config_silver_trips" {
  bucket = var.artifacts_bucket_name
  key    = local.dq_config_key
  source = "${path.module}/../ingestion/dq/rules/silver_trips_dq_config.json"
  etag   = filemd5("${path.module}/../ingestion/dq/rules/silver_trips_dq_config.json")
}

resource "aws_s3_object" "dq_gate_script" {
  bucket = var.artifacts_bucket_name
  key    = local.dq_gate_script_key
  source = "${path.module}/../ingestion/jobs/dq_gate.py"
  etag   = filemd5("${path.module}/../ingestion/jobs/dq_gate.py")
}

resource "aws_s3_object" "crz_zones_seed" {
  bucket = var.artifacts_bucket_name
  key    = local.crz_zones_seed_key
  source = "${path.module}/../seeds/crz_zones.csv"
  etag   = filemd5("${path.module}/../seeds/crz_zones.csv")
}

resource "aws_glue_data_quality_ruleset" "silver_trips" {
  name        = "${var.project_name}-${var.environment}-silver-trips-dq"
  description = "T-103 DQ-01 through DQ-07 for silver.trips"
  ruleset     = file("${path.module}/../ingestion/dq/rules/silver_trips_dq.dqdl")
}

resource "aws_glue_job" "dq_gate_silver_trips" {
  name              = "${var.project_name}-${var.environment}-dq-gate-silver-trips"
  role_arn          = module.iam.pipeline_role_arn
  glue_version      = "5.1"
  worker_type       = "G.1X"
  number_of_workers = 2
  timeout           = 20
  max_retries       = 0

  command {
    name            = "glueetl"
    script_location = "s3://${var.artifacts_bucket_name}/${aws_s3_object.dq_gate_script.key}"
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

    "--catalog"              = "glue_catalog"
    "--silver_db"            = var.silver_glue_database_name
    "--dq_rules_path"        = "s3://${var.artifacts_bucket_name}/${aws_s3_object.dq_rules_silver_trips.key}"
    "--dq_config_path"       = "s3://${var.artifacts_bucket_name}/${aws_s3_object.dq_config_silver_trips.key}"
    "--crz_zones_path"       = "s3://${var.artifacts_bucket_name}/${aws_s3_object.crz_zones_seed.key}"
    "--dq_results_s3_prefix" = "s3://${var.artifacts_bucket_name}/glue/dq-results/silver_trips/"
  }

  execution_property {
    max_concurrent_runs = 1
  }

  tags = local.common_tags
}
