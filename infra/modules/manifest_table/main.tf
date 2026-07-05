resource "aws_dynamodb_table" "manifest" {
  name         = "${var.name_prefix}-ingestion-manifest"
  billing_mode = "PAY_PER_REQUEST"

  hash_key  = "service"
  range_key = "year_month"

  attribute {
    name = "service"
    type = "S"
  }

  attribute {
    name = "year_month"
    type = "S"
  }

  server_side_encryption {
    enabled = true
  }

  point_in_time_recovery {
    enabled = true
  }
}