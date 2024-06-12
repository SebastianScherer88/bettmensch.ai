locals {
  name            = "bettmensch-ai"
  cluster_version = "1.29"
  region          = "us-east-1"

  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)
  oidc_issuer_url = try(replace("${module.eks.cluster_oidc_issuer_url}", "https://", ""), "")

  tags = {
    project    = local.name
    github = "https://github.com/SebastianScherer88/bettmensch.ai"
  }
}