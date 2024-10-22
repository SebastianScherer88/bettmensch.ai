data "aws_iam_policy_document" "ebs_csi_assume_policy" {
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
        "system:serviceaccount:kube-system:ebs-csi-controller-sa",
        "system:serviceaccount:kube-system:ebs-csi-node-sa",
        ]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_issuer_url}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ebs_csi" {
  name               = "bettmensch-ai-ebs-csi-role"
  assume_role_policy = data.aws_iam_policy_document.ebs_csi_assume_policy.json
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ebs_csi" {

  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy" # reduce this
  role       = aws_iam_role.ebs_csi.name
}