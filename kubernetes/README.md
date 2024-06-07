# Setup

The below setup was successfully tested on an AWS EC2 `t2.medium` instance.

1. Install docker by running `1_install_docker.sh`
2. Install `kind` by running `2_install_kind.sh`.
3. Create a `kind` cluster by running `3_create_kind_cluster.sh`.
4. Follow [the `Argo Workflows` quick start docs](https://argo-workflows.readthedocs.io/en/latest/quick-start/#quick-start).
  - 4.1 Deploy the CRDs and non-production resources via yaml file by running `4_deploy_argo_workflows.sh`
5. Install the argo CLI by running `5_install_argo_cli.sh`

# Submitting a workflow

After completing the setup, you can try submitting a workflow. 

1. Run `6_submit_argo_workflow.sh`

To check on the cluster state, jobs, serve applications etc run the following dashboard port forwarding in a separate terminal:
`port_forward_argo_dashboard.sh`

Make sure you connect using `https` when accessing on local; this will work:

```bash
curl https://localhost:2746
```

But this will throw server side errors as per documentation:

```bash
cur http://localhost:2746
```