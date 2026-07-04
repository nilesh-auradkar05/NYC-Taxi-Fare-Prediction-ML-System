locals {
  zones = toset(["bronze", "silver", "gold", "artifacts"])
}

resource "aws_s3_bucket" "lakehouse" {
  for_each = local.zones

  bucket        = "${var.name_prefix}-${each.key}-${var.account_id}"
  force_destroy = var.force_destroy
}

resource "aws_s3_bucket_public_access_block" "lakehouse" {
  for_each = aws_s3_bucket.lakehouse

  bucket = each.value.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "lakehouse" {
  for_each = aws_s3_bucket.lakehouse

  bucket = each.value.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lakehouse" {
  for_each = aws_s3_bucket.lakehouse

  bucket = each.value.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }

    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "lakehouse" {
  for_each = aws_s3_bucket.lakehouse

  bucket = each.value.id

  rule {
    id     = "expire-incomplete-multipart-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "transition-noncurrent-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }
  }
}