locals {
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = {
    Project     = "NYC Mobility Data Decision Platform"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Ticket      = "T-004"
    Owner       = "Nilesh"
  }
}

module "s3_lakehouse" {
  source = "./modules/s3_lakehouse"

  name_prefix   = local.name_prefix
  account_id    = data.aws_caller_identity.current.account_id
  force_destroy = var.force_destroy_buckets
}

module "glue_catalog" {
  source = "./modules/glue_catalog"

  database_names = {
    bronze = "nyc_taxi_bronze"
    silver = "nyc_taxi_silver"
    gold   = "nyc_taxi_gold"
  }
}

module "sns" {
  source = "./modules/sns"

  name_prefix        = local.name_prefix
  budget_alert_email = var.budget_alert_email
}

module "iam" {
  source = "./modules/iam"

  name_prefix = local.name_prefix
  aws_region  = data.aws_region.current.region
  account_id  = data.aws_caller_identity.current.account_id

  bronze_bucket_arn    = module.s3_lakehouse.bucket_arns["bronze"]
  silver_bucket_arn    = module.s3_lakehouse.bucket_arns["silver"]
  gold_bucket_arn      = module.s3_lakehouse.bucket_arns["gold"]
  artifacts_bucket_arn = module.s3_lakehouse.bucket_arns["artifacts"]

  bronze_bucket_name    = module.s3_lakehouse.bucket_names["bronze"]
  silver_bucket_name    = module.s3_lakehouse.bucket_names["silver"]
  gold_bucket_name      = module.s3_lakehouse.bucket_names["gold"]
  artifacts_bucket_name = module.s3_lakehouse.bucket_names["artifacts"]

  glue_database_names      = module.glue_catalog.database_names
  pipeline_alert_topic_arn = module.sns.pipeline_alert_topic_arn
}

module "athena" {
  source = "./modules/athena"

  name_prefix      = local.name_prefix
  artifacts_bucket = module.s3_lakehouse.bucket_names["artifacts"]

  engineering_scan_cutoff_bytes = var.engineering_athena_scan_cutoff_bytes
  agent_scan_cutoff_bytes       = var.agent_athena_scan_cutoff_bytes
}

module "budget" {
  source = "./modules/budget"

  name_prefix        = local.name_prefix
  budget_alert_email = var.budget_alert_email
  monthly_limit_usd  = "40"
}
