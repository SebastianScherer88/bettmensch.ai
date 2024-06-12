################################################################################
# Karpenter
################################################################################

# karpenter controller cant create service linked role for spot, but wont try to do so if the 
# resource already exists before trying to provision spot instances; solution as per
# https://github.com/terraform-aws-modules/terraform-aws-eks/issues/2565
resource "aws_iam_service_linked_role" "AWSServiceRoleForEC2Spot" {
  aws_service_name = "spot.amazonaws.com"
}

module "karpenter" {
  source = "terraform-aws-modules/eks/aws//modules/karpenter"

  cluster_name = module.eks.cluster_name

  # EKS Fargate currently does not support Pod Identity
  enable_irsa            = true
  irsa_oidc_provider_arn = module.eks.oidc_provider_arn

  # Used to attach additional IAM policies to the Karpenter node IAM role
  node_iam_role_additional_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }

  tags = local.tags
}

module "karpenter_disabled" {
  source = "terraform-aws-modules/eks/aws//modules/karpenter"

  create = false
}

 # karpenter installation
data "aws_ecrpublic_authorization_token" "token" {
  provider = aws.north_virginia
}
resource "helm_release" "karpenter" {
  namespace           = "karpenter"
  create_namespace    = true
  name                = "karpenter"
  repository          = "oci://public.ecr.aws/karpenter"
  repository_username = data.aws_ecrpublic_authorization_token.token.user_name
  repository_password = data.aws_ecrpublic_authorization_token.token.password
  chart               = "karpenter"
  version             = "0.35.1"
  wait                = false

  values = [
    <<-EOT
    settings:
      clusterName: ${module.eks.cluster_name}
      clusterEndpoint: ${module.eks.cluster_endpoint}
      interruptionQueue: ${module.karpenter.queue_name}
    serviceAccount:
      annotations:
        eks.amazonaws.com/role-arn: ${module.karpenter.iam_role_arn}
    EOT
  ]

}

# karpenter node class
resource "kubectl_manifest" "karpenter_node_class" {
  yaml_body = <<-YAML
    apiVersion: karpenter.k8s.aws/v1beta1
    kind: EC2NodeClass
    metadata:
      name: default
    spec:
      amiFamily: AL2
      role: ${module.karpenter.node_iam_role_name}
      subnetSelectorTerms:
        - tags:
            karpenter.sh/discovery: ${module.eks.cluster_name}
      securityGroupSelectorTerms:
        - tags:
            karpenter.sh/discovery: ${module.eks.cluster_name}
      tags:
        karpenter.sh/discovery: ${module.eks.cluster_name}
  YAML

  depends_on = [
    helm_release.karpenter
  ]
}

# karpenter nodepool
data "kubectl_file_documents" "karpenter_node_pool" {
    content = file("../../kubernetes/manifests/karpenter/karpenter-nodepool.yaml")
}

resource "kubectl_manifest" "karpenter_node_pool" {
  for_each  = data.kubectl_file_documents.karpenter_node_pool.manifests
  yaml_body = each.value
  
  depends_on = [
    kubectl_manifest.karpenter_node_class
  ]
}

################################################################################
# Argo Workflows
################################################################################

# argo workflows installation
data "kubectl_file_documents" "argo_workflows" {
    content = file("../../kubernetes/manifests/argo_workflows/argo-workflows-installation.yaml")
}

resource "kubectl_manifest" "argo_workflows" {
    for_each  = data.kubectl_file_documents.argo_workflows.manifests
    yaml_body = each.value

    depends_on = [
    kubectl_manifest.karpenter_node_pool
  ]
}

# argo artifact repository configuration
resource "kubectl_manifest" "argo_workflows_artifact_repository" {
  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      annotations:
        workflows.argoproj.io/default-artifact-repository: bettmensch-ai-artifact-repository
      name: artifact-repositories
      namespace: argo
    data:
      bettmensch-ai-artifact-repository: |
        s3:
          bucket: ${resource.aws_s3_bucket.artifact_repository.id}
          endpoint: s3.amazonaws.com
          insecure: true
  YAML

  depends_on = [
    kubectl_manifest.karpenter_node_pool
  ]
}

################################################################################
# Volcano
################################################################################

# volcano installation
data "kubectl_file_documents" "volcano" {
    content = file("../../kubernetes/manifests/volcano/volcano-installation.yaml")
}

resource "kubectl_manifest" "volcano" {
    for_each  = data.kubectl_file_documents.volcano.manifests
    yaml_body = each.value

depends_on = [
    kubectl_manifest.karpenter_node_pool
  ]
}