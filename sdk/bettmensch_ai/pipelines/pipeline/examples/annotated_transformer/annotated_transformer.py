from bettmensch_ai.pipelines import Pipeline, as_pipeline
from bettmensch_ai.pipelines.component.examples.annotated_transformer import (
    get_tokenizers_and_vocabularies_factory,
    train_transformer_factory,
)
from bettmensch_ai.pipelines.constants import ARGO_NAMESPACE, COMPONENT_IMAGE
from bettmensch_ai.pipelines.io import InputParameter
from hera.workflows import EmptyDirVolume


def get_train_transformer_pipeline(
    name: str = "test-train-pipeline-1n-1p-",
    n_nodes: int = 1,
    n_proc_per_node: int = 1,
) -> Pipeline:
    @as_pipeline(name, ARGO_NAMESPACE, True)
    def train_transformer_pipeline(
        dataset: InputParameter = "multi30k",
        source_language: InputParameter = "de",
        target_language: InputParameter = "en",
        batch_size: InputParameter = 32,
        num_epochs: InputParameter = 8,
        accum_iter: InputParameter = 10,
        base_lr: InputParameter = 1.0,
        max_padding: InputParameter = 72,
        warmup: InputParameter = 3000,
    ):

        # preprocessing component
        get_tokenizers_and_vocabularies = (
            get_tokenizers_and_vocabularies_factory(
                "preprocess",
                hera_template_kwargs={
                    "image": COMPONENT_IMAGE.annotated_transformer.value
                },
                dataset=dataset,
                source_language=source_language,
                target_language=target_language,
                max_padding=max_padding,
            )
            .set_memory("1000Mi")
            .set_cpu(1)
        )

        # training component
        hera_template_kwargs = {
            "image": COMPONENT_IMAGE.annotated_transformer.value,
        }

        if n_proc_per_node > 1:
            # enable same node gpu comms as per
            # https://hera.readthedocs.io/en/latest/examples/workflows/...
            # ...use-cases/fine_tune_llama/
            hera_template_kwargs["volumes"] = [
                EmptyDirVolume(
                    name="gpu-comm", size_limit="15Gi", mount_path="/dev/shm"
                )
            ]

        if n_nodes > 1:
            # force different k8s nodes where applicable to ensure a multi-node
            # run is genuinely multi-node not just at the torchrun level but
            # also at the k8s level. For testing coverage purposes only
            hera_template_kwargs[
                "pod_spec_patch"
            ] = """topologySpreadConstraints:
- maxSkew: 1
  topologyKey: kubernetes.io/hostname
  whenUnsatisfiable: DoNotSchedule
  labelSelector:
    matchExpressions:
      - { key: torch-node, operator: In, values: ['0','1','2','3','4','5']}"""

        train_transformer = (  # noqa: F841
            train_transformer_factory(
                "train-transformer",
                hera_template_kwargs=hera_template_kwargs,
                n_nodes=n_nodes,
                min_nodes=n_nodes,
                nproc_per_node=n_proc_per_node,
                dataset=dataset,
                source_language=source_language,
                target_language=target_language,
                source_tokenizer=get_tokenizers_and_vocabularies.outputs[
                    "source_tokenizer"
                ],
                target_tokenizer=get_tokenizers_and_vocabularies.outputs[
                    "target_tokenizer"
                ],
                vocabularies=get_tokenizers_and_vocabularies.outputs[
                    "vocabularies"
                ],
                batch_size=batch_size,
                num_epochs=num_epochs,
                accum_iter=accum_iter,
                base_lr=base_lr,
                max_padding=max_padding,
                warmup=warmup,
            )
            .set_cpu(3 * n_proc_per_node)  # works with 3 CPUs per process
            .set_gpus(1 * n_proc_per_node)  # 1 GPU per process
            .set_memory(
                f"{int(4 * n_proc_per_node)}Gi"
            )  # works with 4Gi per process
        )

    return train_transformer_pipeline


train_transformer_pipeline_1n_1p = get_train_transformer_pipeline()
train_transformer_pipeline_1n_2p = get_train_transformer_pipeline(
    name="test-train-pipeline-1n-2p-", n_nodes=1, n_proc_per_node=2
)
train_transformer_pipeline_2n_1p = get_train_transformer_pipeline(
    name="test-train-pipeline-2n-1p-", n_nodes=2, n_proc_per_node=1
)
train_transformer_pipeline_2n_2p = get_train_transformer_pipeline(
    name="test-train-pipeline-2n-2p-", n_nodes=2, n_proc_per_node=2
)
