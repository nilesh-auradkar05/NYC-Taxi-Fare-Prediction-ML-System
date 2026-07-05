locals {
  lakehouse_bucket_arns = [
    var.bronze_bucket_arn,
    var.silver_bucket_arn,
    var.gold_bucket_arn,
    var.artifacts_bucket_arn,
  ]

  lakehouse_object_arns = [
    "${var.bronze_bucket_arn}/*",
    "${var.silver_bucket_arn}/*",
    "${var.gold_bucket_arn}/*",
    "${var.artifacts_bucket_arn}/*",
  ]

  glue_catalog_arn = "arn:aws:glue:${var.aws_region}:${var.account_id}:catalog"

  glue_database_arns = [
    "arn:aws:glue:${var.aws_region}:${var.account_id}:database/${var.glue_database_names.bronze}",
    "arn:aws:glue:${var.aws_region}:${var.account_id}:database/${var.glue_database_names.silver}",
    "arn:aws:glue:${var.aws_region}:${var.account_id}:database/${var.glue_database_names.gold}",
  ]

  gold_glue_resources = [
    local.glue_catalog_arn,
    "arn:aws:glue:${var.aws_region}:${var.account_id}:database/${var.glue_database_names.gold}",
    "arn:aws:glue:${var.aws_region}:${var.account_id}:table/${var.glue_database_names.gold}/*",
  ]
}

data "aws_iam_policy_document" "pipeline_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type = "Service"
      identifiers = [
        "lambda.amazonaws.com",
        "glue.amazonaws.com",
        "states.amazonaws.com",
        "events.amazonaws.com"
      ]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "pipeline_execution" {
  name               = "${var.name_prefix}-pipeline-execution"
  assume_role_policy = data.aws_iam_policy_document.pipeline_assume_role.json
}

data "aws_iam_policy_document" "pipeline_permissions" {
  statement {
    sid     = "LakehouseBucketList"
    effect  = "Allow"
    actions = ["s3:ListBucket"]

    resources = local.lakehouse_bucket_arns
  }

  statement {
    sid    = "LakehouseObjectReadWrite"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts"
    ]

    resources = local.lakehouse_object_arns
  }

  statement {
    sid    = "GlueCatalogProjectDatabases"
    effect = "Allow"
    actions = [
      "glue:GetDatabase",
      "glue:GetDatabases",
      "glue:CreateTable",
      "glue:UpdateTable",
      "glue:DeleteTable",
      "glue:GetTable",
      "glue:GetTables",
      "glue:GetPartition",
      "glue:GetPartitions",
      "glue:CreatePartition",
      "glue:UpdatePartition",
      "glue:DeletePartition",
      "glue:BatchCreatePartition",
      "glue:BatchDeletePartition",
      "glue:BatchGetPartition"
    ]

    resources = concat(
      [local.glue_catalog_arn],
      local.glue_database_arns,
      [
        "arn:aws:glue:${var.aws_region}:${var.account_id}:table/${var.glue_database_names.bronze}/*",
        "arn:aws:glue:${var.aws_region}:${var.account_id}:table/${var.glue_database_names.silver}/*",
        "arn:aws:glue:${var.aws_region}:${var.account_id}:table/${var.glue_database_names.gold}/*"
      ]
    )
  }

  statement {
    sid    = "AthenaEngineeringExecution"
    effect = "Allow"
    actions = [
      "athena:StartQueryExecution",
      "athena:StopQueryExecution",
      "athena:GetQueryExecution",
      "athena:GetQueryResults",
      "athena:GetWorkGroup",
      "athena:ListQueryExecutions"
    ]

    resources = [
      "arn:aws:athena:${var.aws_region}:${var.account_id}:workgroup/engineering"
    ]
  }

  statement {
    sid    = "CloudWatchLogsForLambdaGlue"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]

    resources = ["arn:aws:logs:${var.aws_region}:${var.account_id}:*"]
  }

  statement {
    sid       = "PublishPipelineAlerts"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [var.pipeline_alert_topic_arn]
  }

  statement {
    sid    = "ManifestTableReadWrite"
    effect = "Allow"

    actions = [
      "dynamodb:DescribeTable",
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem"
    ]

    resources = [var.manifest_table_arn]
  }
}

resource "aws_iam_policy" "pipeline_permissions" {
  name        = "${var.name_prefix}-pipeline-permissions"
  description = "Least-privilege permissions for NYC Mobility pipeline execution."
  policy      = data.aws_iam_policy_document.pipeline_permissions.json
}

resource "aws_iam_role_policy_attachment" "pipeline_permissions" {
  role       = aws_iam_role.pipeline_execution.name
  policy_arn = aws_iam_policy.pipeline_permissions.arn
}

data "aws_iam_policy_document" "agent_readonly_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${var.account_id}:root"]
    }

    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "agent_readonly" {
  name               = "${var.name_prefix}-agent-readonly"
  assume_role_policy = data.aws_iam_policy_document.agent_readonly_assume_role.json
}

data "aws_iam_policy_document" "agent_readonly_permissions" {
  statement {
    sid     = "ReadGoldBucket"
    effect  = "Allow"
    actions = ["s3:ListBucket"]

    resources = [var.gold_bucket_arn]
  }

  statement {
    sid    = "ReadGoldObjects"
    effect = "Allow"
    actions = [
      "s3:GetObject"
    ]

    resources = ["${var.gold_bucket_arn}/*"]
  }

  statement {
    sid     = "ListArtifactsBucketForAthenaResults"
    effect  = "Allow"
    actions = ["s3:ListBucket"]

    resources = [var.artifacts_bucket_arn]
  }

  statement {
    sid    = "AthenaResultObjects"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject"
    ]

    resources = [
      "${var.artifacts_bucket_arn}/athena-results/agent-readonly/*"
    ]
  }

  statement {
    sid    = "ReadGoldGlueCatalog"
    effect = "Allow"
    actions = [
      "glue:GetDatabase",
      "glue:GetDatabases",
      "glue:GetTable",
      "glue:GetTables",
      "glue:GetPartition",
      "glue:GetPartitions"
    ]

    resources = local.gold_glue_resources
  }

  statement {
    sid    = "AgentReadonlyAthena"
    effect = "Allow"
    actions = [
      "athena:StartQueryExecution",
      "athena:StopQueryExecution",
      "athena:GetQueryExecution",
      "athena:GetQueryResults",
      "athena:GetWorkGroup",
      "athena:ListQueryExecutions"
    ]

    resources = [
      "arn:aws:athena:${var.aws_region}:${var.account_id}:workgroup/agent-readonly"
    ]
  }
}

resource "aws_iam_policy" "agent_readonly_permissions" {
  name        = "${var.name_prefix}-agent-readonly-permissions"
  description = "Read-only permissions for the analyst agent over gold data through Athena."
  policy      = data.aws_iam_policy_document.agent_readonly_permissions.json
}

resource "aws_iam_role_policy_attachment" "agent_readonly_permissions" {
  role       = aws_iam_role.agent_readonly.name
  policy_arn = aws_iam_policy.agent_readonly_permissions.arn
}