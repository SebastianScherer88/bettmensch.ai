resource "aws_s3_bucket" "artifact_repository" {
  bucket = "bettmensch-ai-artifact-repository"
  tags = local.tags
}

data "aws_iam_policy_document" "artifact_repository_assume_policy" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type = "Federated"
      identifiers = [
        module.eks.oidc_provider_arn
      ]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_issuer_url}:sub"
      values   = [
        "system:serviceaccount:argo:argo-server",
        "system:serviceaccount:argo:default",
        ]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_issuer_url}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "artifact_repository" {
  name               = "bettmensch-ai-pipelines-artifact-role"
  assume_role_policy = data.aws_iam_policy_document.artifact_repository_assume_policy.json
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "artifact_repository" {

  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess" # reduce this
  role       = aws_iam_role.artifact_repository.name
}