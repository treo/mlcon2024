from xai_components.base import InArg, OutArg, Component, xai_component
import re

@xai_component
class ExtractResponse(Component):
    """A component that extracts the content within <response> tags from a string containing <thinking> and <response> tags.

    ##### inPorts:
    - input_text: a String port containing text with <thinking> and <response> tags.

    ##### outPorts:
    - response_text: a String port containing the content within <response> tags.
    """
    input_text: InArg[str]
    response_text: OutArg[str]

    def execute(self, ctx) -> None:
        input_text = self.input_text.value

        # Use regular expressions to extract text within <response> tags
        response_content = re.search(r'<response>(.*?)</response>', input_text, re.DOTALL)
        
        if response_content:
            self.response_text.value = response_content.group(1)
        else:
            self.response_text.value = input_text
