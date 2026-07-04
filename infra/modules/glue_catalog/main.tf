resource "aws_glue_catalog_database" "bronze" {
  name        = var.database_names.bronze
  description = "Bronze raw TLC trip data catalog."
}

resource "aws_glue_catalog_database" "silver" {
  name        = var.database_names.silver
  description = "Silver conformed trip-level Iceberg tables."
}

resource "aws_glue_catalog_database" "gold" {
  name        = var.database_names.gold
  description = "Gold marts for causal analysis, forecasting, agent, and BI."
}