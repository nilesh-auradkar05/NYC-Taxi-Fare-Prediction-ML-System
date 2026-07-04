output "bucket_names" {
  description = "Map of lakehouse zone to bucket name."
  value = {
    for zone, bucket in aws_s3_bucket.lakehouse : zone => bucket.bucket
  }
}

output "bucket_arns" {
  description = "Map of lakehouse zone to bucket ARN."
  value = {
    for zone, bucket in aws_s3_bucket.lakehouse : zone => bucket.arn
  }
}