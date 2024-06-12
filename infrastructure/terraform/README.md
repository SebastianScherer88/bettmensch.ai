# Overview

To provision 
- the S3 bucket for the Argo Workflows artifact repository
- Karpenter required infrastructure (IAM, message queues, etc.)
- a working EKS cluster
- the configured Karpenter, Argo Workflows & Volcano kubernetes installations 
    on the cluster

and then configure your `kubectl` to point towards the EKS cluster, run:

```bash
terraform init
terraform plan
terraform apply -auto-approve
aws eks --region us-east-1 update-kubeconfig --name bettmensch-ai
```