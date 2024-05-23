# :hotel: Welcome to Bettmensch.AI

![bettmensch.ai logo](image/logo_transparent.png)

:factory: Bettmensch.AI is a Kubernetes native open source platform for GitOps based ML workloads that allows for tight CI and CD integrations.

# Features (under active development )

## :computer: Dashboard

![bettmensch.ai](image/dashboard_0_home.JPG)

:eyes: A dashboard for *monitoring* all workloads running on the platform.

:open_hands: To actively *manage* `Pipeline`s, `Flow`s, please see the respective documentation of `bettmensch.ai` SDK.

## :twisted_rightwards_arrows: `Pipelines & Flows`

![bettmensch.ai](image/dashboard_1_pipelines.JPG)

`bettmensch.ai` comes with a python SDK for defining and executing distributed (ML) workloads by leveraging the [`ArgoWorkflows`](https://argoproj.github.io/workflows/) framework and the official [`hera`](https://github.com/argoproj-labs/hera) library.

```
from bettmensch_ai.arguments import (
    ComponentInput,
    ComponentOutput,
    PipelineInput,
)
from bettmensch_ai.component import component
from bettmensch_ai.pipeline import pipeline

@component
def add(
    a: ComponentInput, b: ComponentInput, sum: ComponentOutput = None
) -> None:

    sum.assign(a + b)

@component
def multiply(
    a: ComponentInput, b: ComponentInput, product: ComponentOutput = None
) -> None:

    product.assign(a * b)

@pipeline("test-pipeline", "argo", True)
def a_plus_bc_plus_2b(
    a: PipelineInput = 1, b: PipelineInput = 2, c: PipelineInput = 3
):

    b_c = multiply(
        "bc",
        a=b,
        b=c,
    )

    two_b = multiply(
        "b2",
        a=b,
        b=ComponentInput(name="two", value=2),
    )

    a_plus_bc = add(
        "a-plus-bc",
        a=a,
        b=b_c.outputs["product"],
    )

    result_a_plus_bc_plus_2b = add(
        "result-a-plus-bc-plus-2b",
        a=a_plus_bc.outputs["sum"],
        b=two_b.outputs["product"],
    )

a_plus_bc_plus_2b.export()
a_plus_bc_plus_2b.register()
a_plus_bc_plus_2b.run(a=3, b=2, c=1)
```

![bettmensch.ai](image/dashboard_2_flows.JPG)

## :books: `Models`

![bettmensch.ai models](image/dashboard_3_models.JPG)

Coming soon.

## :rocket: `Servers`

![bettmensch.ai servers](image/dashboard_4_servers.JPG)

Coming soon.