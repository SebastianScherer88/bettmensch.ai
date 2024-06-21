# Overview

To provision 
- the S3 bucket for the Argo Workflows artifact repository
- Karpenter required infrastructure (IAM, message queues, etc.)
- a working EKS cluster
- the configured Karpenter, Argo Workflows & Volcano kubernetes installations 
    on the cluster,

run:

```bash
terraform init
terraform plan
terraform apply -auto-approve
```

To configure your kubectl to point towards the EKS cluster, run:

```bash
aws eks --region us-east-2 update-kubeconfig --name bettmensch-ai
```

To port forward the argo server to your local port `2746` so you can access the argo dashboard and start
submitting & running pipelines, run

```bash
bash ../../kubernetes/port_forward_argo_server.sh
```