from bettmensch_ai import InputParameter, OutputParameter, component, pipeline


def test_parameter_pipeline():
    @component
    def add(
        a: InputParameter, b: InputParameter, sum: OutputParameter = None
    ) -> None:

        sum.assign(a + b)

    @component
    def multiply(
        a: InputParameter, b: InputParameter, product: OutputParameter = None
    ) -> None:

        product.assign(a * b)

    @pipeline("test-parameter-pipeline", "argo", True)
    def a_plus_bc_plus_2b(
        a: InputParameter = 1, b: InputParameter = 2, c: InputParameter = 3
    ):

        b_c = multiply(
            "bc",
            a=b,
            b=c,
        )

        two_b = multiply(
            "b2",
            a=b,
            b=InputParameter(name="two", value=2),
        )

        a_plus_bc = add(
            "a-plus-bc",
            a=a,
            b=b_c.outputs["product"],
        )

        result = add(
            "a-plus-bc-plus-2b",
            a=a_plus_bc.outputs["sum"],
            b=two_b.outputs["product"],
        )

    print(f"Pipeline type: {type(a_plus_bc_plus_2b)}")

    a_plus_bc_plus_2b.export("./outputs")
    # a_plus_bc_plus_2b.register()
    # a_plus_bc_plus_2b.run(a=3, b=2, c=1)
