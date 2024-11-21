from bettmensch_ai.pipelines.component import (
    TorchDDPComponentInlineScriptRunner,
)


def test__get_invocation_script_portion(test_mock_script):

    script_runner = TorchDDPComponentInlineScriptRunner()
    invocation_script_portion = script_runner._get_invocation_script_portion(
        test_mock_script
    )

    assert (
        invocation_script_portion
        == "\nfrom torch.distributed.elastic.multiprocessing.errors import record\n\ntest_function=record(test_function)\n\nfrom bettmensch_ai.pipelines.component import as_torch_ddp\n\ntorch_ddp_decorator=as_torch_ddp()\n\ntorch_ddp_function=torch_ddp_decorator(test_function)\n\n\ntorch_ddp_function(a,b,c,d,a_out,b_out)"  # noqa: E501
    )


def test_generate_source(test_mock_script):

    script_runner = TorchDDPComponentInlineScriptRunner()
    generated_source = script_runner.generate_source(test_mock_script)

    assert (
        generated_source
        == "\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\ntry: b = json.loads(r'''{{inputs.parameters.b}}''')\nexcept: b = r'''{{inputs.parameters.b}}'''\ntry: c = json.loads(r'''{{inputs.parameters.c}}''')\nexcept: c = r'''{{inputs.parameters.c}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import InputArtifact\nd = InputArtifact(\"d\")\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\na_out = OutputParameter(\"a_out\")\n\nfrom bettmensch_ai.pipelines.io import OutputArtifact\nb_out = OutputArtifact(\"b_out\")\n\ndef test_function(a: InputParameter, b: InputParameter, c: InputParameter, d: InputArtifact, a_out: OutputParameter, b_out: OutputArtifact):\n    pass\n\nfrom torch.distributed.elastic.multiprocessing.errors import record\n\ntest_function=record(test_function)\n\nfrom bettmensch_ai.pipelines.component import as_torch_ddp\n\ntorch_ddp_decorator=as_torch_ddp()\n\ntorch_ddp_function=torch_ddp_decorator(test_function)\n\n\ntorch_ddp_function(a,b,c,d,a_out,b_out)"  # noqa: E501
    )
