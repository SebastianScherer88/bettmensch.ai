from bettmensch_ai.pipelines.component.base.inline_script_runner import (
    BaseComponentInlineScriptRunner,
)
from bettmensch_ai.pipelines.component.standard.script import (
    BettmenschAIStandardScript,
)
from hera.shared import global_config


class ComponentInlineScriptRunner(BaseComponentInlineScriptRunner):

    """
    A customised version of the InlineScriptConstructor that implements a
    modified `_get_param_script_portion` and `generate_source` methods to
    ensure proper handling of the SDK's I/O objects at runtime.
    """

    pass


global_config.set_class_defaults(
    BettmenschAIStandardScript, constructor=ComponentInlineScriptRunner()
)
