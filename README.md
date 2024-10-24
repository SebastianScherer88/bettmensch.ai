# :hotel: Welcome to Bettmensch.AI

![bettmensch.ai logo](image/logo_transparent.png)

Bettmensch.AI is a Kubernetes native open source platform for GitOps based ML workloads that allows for tight CI and CD integrations.

![docker](https://github.com/SebastianScherer88/bettmensch.ai/actions/workflows/docker.yaml/badge.svg)
![unit tests](https://github.com/SebastianScherer88/bettmensch.ai/actions/workflows/unit-test.yaml/badge.svg)
![integration tests](https://github.com/SebastianScherer88/bettmensch.ai/actions/workflows/integration-test.yaml/badge.svg)
![k8s tests](https://github.com/SebastianScherer88/bettmensch.ai/actions/workflows/k8s-test.yaml/badge.svg)

# Setup

## :bridge_at_night: AWS Infrastructure & Kubernetes
To provision 
- the S3 bucket for the Argo Workflows artifact repository
- Karpenter required infrastructure (IAM, message queues, etc.)
- a working EKS cluster
- the configured Karpenter, Argo Workflows & Volcano kubernetes installations 
    on the cluster,

```bash
make platform.up
```

To port forward to 
- the `ArgoWorkflow` server running on EKS and
- the `Mlflow` server running on EKS,

run:

```bash
make platform.connect
```

When you're done, you can tear down the stack by running

```bash
make platform.down
```

## :computer: Dashboard

To build the `bettmensch.ai`'s custom dashboard's docker image, run:

```bash
make dashboard.build
```

To run the dashboard, run:

```bash
make dashboard.run
```

See the `docker` section for more details.

## :books: Python SDK installation

To install the python library `bettmensch_ai` with `torch-pipelines` support,
 run 
 
```bash
make sdk.install EXTRAS=torch-pipelines
```

from the repository's top directory.

You can now start authoring `Pipeline`s and start submitting `Flow`s and 
start monitoring them on both the `ArgoWorkflow` as well as the `bettmensch.ai`
dashboards.

## :wrench: Running tests

To run unit tests for the python library, run

```bash
make sdk.test SUITE=unit
```

To run integration tests for the python library, run

```bash
make sdk.test SUITE=integration
```

To run K8s tests for the python library (requires a running and connected
bettmensch.ai platform), run

```bash
make sdk.test SUITE=k8s
```

# Features (under active development )

## :computer: Dashboard

![bettmensch.ai](image/dashboard_0_home.JPG)

:eyes: A dashboard for *monitoring* all workloads running on the platform.

:open_hands: To actively *manage* `Pipeline`s, `Flow`s, please see the 
respective documentation of `bettmensch.ai` SDK.

## :twisted_rightwards_arrows: `Pipelines & Flows`

`bettmensch.ai` comes with a python SDK for defining and executing distributed
 (ML) workloads by leveraging the 
 [`ArgoWorkflows`](https://argoproj.github.io/workflows/) framework and the
  official [`hera`](https://github.com/argoproj-labs/hera) library.

The `io` module implements the classes implementing the transfer of inputs and
 outputs between a workfload's components.

Using `InputParameter` and `OutputParameter` for `int`, `float` or `str` type 
data:

```python
from bettmensch_ai.io import InputParameter, OutputParameter
from bettmensch_ai.components import component
from bettmensch_ai.pipelines import pipeline

@component
def add(
    a: InputParameter = 1,
    b: InputParameter = 2,
    sum: OutputParameter = None,
) -> None:

    sum.assign(a + b)

@pipeline("test-parameter-pipeline", "argo", True)
def a_plus_b_plus_2(a: InputParameter = 1, b: InputParameter = 2) -> None:
    a_plus_b = add(
        "a-plus-b",
        a=a,
        b=b,
    )

    a_plus_b_plus_2 = add(
        "a-plus-b-plus-2",
        a=a_plus_b.outputs["sum"],
        b=InputParameter("two", 2),
    )

a_plus_b_plus_2.export(test_output_dir)
a_plus_b_plus_2.register()
a_plus_b_plus_2.run(a=3, b=2)
```

Using `InputArtifact` and `OutputArtifact` for all other types of data, 
leveraging AWS's `S3` storage service:

```python
from bettmensch_ai.io import InputArtifact, OutputArtifact
from bettmensch_ai.components import component
from bettmensch_ai.pipelines import pipeline

@component
def convert_to_artifact(
    a_param: InputParameter,
    a_art: OutputArtifact = None,
) -> None:

    with open(a_art.path, "w") as a_art_file:
        a_art_file.write(str(a_param))

@component
def show_artifact(a: InputArtifact) -> None:

    with open(a.path, "r") as a_art_file:
        a_content = a_art_file.read()

    print(f"Content of input artifact a: {a_content}")

@pipeline("test-artifact-pipeline", "argo", True)
def parameter_to_artifact(
    a: InputParameter = "Param A",
) -> None:
    convert = convert_to_artifact(
        "convert-to-artifact",
        a_param=a,
    )

    show = show_artifact(
        "show-artifact",
        a=convert.outputs["a_art"],
    )

parameter_to_artifact.export(test_output_dir)
parameter_to_artifact.register()
parameter_to_artifact.run(a="Test value A")
```

**NOTE**: For more examples (including cross K8s node CPU and GPU `torch.distributed` 
processes), see this repository's [integration `test`](./sdk/test/integration/)
 and [k8s `test`](./sdk/test/k8s/) sections.

The submitted pipelines can be viewed on the dashboard's `Pipelines` section:

![bettmensch.ai pipelines](image/dashboard_1_pipelines.JPG)

The executed flows can be viewed on the dashboard's `Flows` section:

![bettmensch.ai flows](image/dashboard_2_flows.JPG)

The following sequence diagram illustrates how the creation, registration and
 running of `Pipeline`'s is supported by the infrastructure stack initiated in
 the `Setup` section:
![BettmenschAI - Sequence diagram](https://github.com/user-attachments/assets/fb930dcc-d856-4224-8a7d-790a85269c73)

## :books: `Models`

![bettmensch.ai models](image/dashboard_3_models.JPG)

Coming soon.

## :rocket: `Servers`

![bettmensch.ai servers](image/dashboard_4_servers.JPG)

Coming soon.

# Credits

This platform makes liberal use of various great open source projects:
- [ArgoWorkflows](https://argoproj.github.io/workflows/): Kubernetes native 
workload orchestration. Also powers the popular
 [Kubeflow Pipelines](https://github.com/kubeflow/pipelines), which inspired 
 the `Pipelines` & `Flows` of this project.
- [hera](https://github.com/argoproj-labs/hera): Official Argo Python SDK for
 defining Workflow(Template)s
- [streamlit](https://streamlit.io/): A python library for designing 
interactive dashboards
  - [streamlit-flow-component](https://github.com/dkapur17/streamlit-flow): A
   [react-flow](https://reactflow.dev/) integration for streamlit
  - [st-pages](https://st-pages.streamlit.app/): A nice streamlit plugin for
   multi-page dashboards
- [mlflow](https://mlflow.org/): ML experiment tracking, model registry and
    serving support
