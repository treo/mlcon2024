from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component, dynalist, secret, SubGraphExecutor

import abc
from collections import deque
from typing import NamedTuple

import json
import os
import requests
import random
import string
import copy

try:
    import openai
except Exception as e:
    pass

# Optional: If using NumpyMemory need numpy and OpenAI
try:
    import numpy as np
except Exception as e:
    pass

# Optional: If using vertexai provider.
try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
except Exception as e:
    pass


def random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


class Memory(abc.ABC):
    def query(self, query: str, n: int) -> list:
        pass

    def add(self, id: str, text: str, metadata: dict) -> None:
        pass


class VectoMemoryImpl(Memory):
    def __init__(self, vs):
        self.vs = vs

    def query(self, query: str, n: int) -> list:
        return self.vs.lookup(query, 'TEXT', n)
    def add(self, id: str, text: str, metadata: dict) -> None:
        from vecto import vecto_toolbelt

        vecto_toolbelt.ingest_text(self.vs, [text], [metadata])



def get_ada_embedding(text):
    s = text.replace("\n", " ")
    return openai.Embedding.create(input=[s], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


class NumpyQueryResult(NamedTuple):
    id: str
    similarity: float
    attributes: dict


class NumpyMemoryImpl(Memory):
    def __init__(self, vectors=None, ids=None, metadata=None):
        self.vectors = vectors
        self.ids = ids
        self.metadata = metadata

    def query(self, query: str, n: int) -> list:
        if self.vectors is None:
            return []
        if isinstance(self.vectors, list) and len(self.vectors) > 1:
            self.vectors = np.vstack(self.vectors)

        top_k = min(self.vectors.shape[0], n)
        query_vector = get_ada_embedding(query)
        similarities = self.vectors @ query_vector
        indices = np.argpartition(similarities, -top_k)[-top_k:]
        return [
            NumpyQueryResult(
                self.ids[i],
                similarities[i],
                self.metadata[i]
            )
            for i in indices
        ]

    def add(self, vector_id: str, text: str, metadata: dict) -> None:
        if isinstance(self.vectors, list) and len(self.vectors) > 1:
            self.vectors = np.vstack(self.vectors)

        if self.vectors is None:
            self.vectors = np.array(get_ada_embedding(text)).reshape((1, -1))
            self.ids = [vector_id]
            self.metadata = [metadata]
        else:
            self.ids.append(vector_id)
            self.vectors = np.vstack([self.vectors, np.array(get_ada_embedding(text))])
            self.metadata.append(metadata)


@xai_component
class AgentNumpyMemory(Component):
    """Creates a local and temporary memory for the agent to store and query information.

    ##### outPorts:
    - memory: The Memory to set on AgentInit
    """
     
    memory: OutArg[Memory]

    def execute(self, ctx) -> None:
        self.memory.value = NumpyMemoryImpl()


class Tool(NamedTuple):
    name: str
    description: str
    inputs: str
    outputs: str


class MutableVariable:
    _fn: any

    def __init__(self):
        self._fn = None
    
    def set_fn(self, fn) -> None:
        self._fn = fn
        
    @property
    def value(self) -> any:
        return self._fn()


@xai_component(type="Start", color="red")
class AgentDefineTool(Component):    
    """Define a tool that the agent can use when it deems necessary.

    This event will be called when the Agent uses this tool.  Perform the tool
    actions and set the output with AgentToolOutput

    ##### inPorts:
    - tool_name: The name of the tool.
    - description: The description of the tool.
    - for_toolbelt: The toolbelt to add the tool to.  If not set, will be added to the default toolbelt.

    ##### outPorts:
    - tool_input: The input for the tool coming from the agent.

    """

    tool_name: InCompArg[str]
    description: InCompArg[str]
    for_toolbelt: InArg[str]
    
    tool_input: OutArg[str]

    
    def init(self, ctx):
        toolbelt = self.for_toolbelt.value if self.for_toolbelt.value is not None else 'default'
        ctx.setdefault('toolbelt_' + toolbelt, {})[self.tool_name.value] = self
        self.tool_ref = InCompArg(None)
        
    
    def execute(self, ctx) -> None:
        other_self = self
        
        class CustomTool(Tool):
            name = other_self.tool_name.value
            description = other_self.description.value
            inputs = ["text"]
            output = ["text"]
            
            def __call__(self, prompt):
                other_self.tool_input.value = prompt
                next = other_self.next
                while next:
                    next = next.do(ctx)
                result = ctx['tool_output']
                ctx['tool_output'] = None
                return result
            
        self.tool_ref.value = CustomTool(
            self.tool_name.value,
            self.description.value,
            ["text"],
            ["text"]
        )

@xai_component(color="red")
class AgentToolOutput(Component):
    """Output the result of the tool to the agent.

    ##### inPorts:
    - results: The results of the tool to be returned to the agent.

    """

    results: InArg[dynalist]
    
    def execute(self, ctx) -> None:
        if len(self.results.value) == 1:
            ctx['tool_output'] = self.results.value[0]



@xai_component
class AgentMakeToolbelt(Component):
    """Create a toolbelt for the agent to use.

    ##### inPorts:
    - name: The name of the toolbelt.

    ##### outPorts:
    - toolbelt_spec: The toolbelt to set on AgentInit

    """
    name: InArg[str]
    toolbelt_spec: OutArg[dict]

    def execute(self, ctx) -> None:
        spec = {}

        toolbelt_name = self.name.value if self.name.value is not None else 'default'
        
        for tool in ctx['toolbelt_' + toolbelt_name]:
            if tool is not None:
                tool_component = ctx['toolbelt_' + toolbelt_name][tool]
                tool_component.execute(ctx)
                spec[tool_component.tool_ref.value.name] = tool_component.tool_ref.value
            
        self.toolbelt_spec.value = spec


@xai_component
class AgentVectoMemory(Component):
    """Creates a memory for the agent to store and query information.
    
    ##### inPorts:
    - api_key: The API key for Vecto.
    - vector_space: The name of the vector space to use.
    - initialize: Whether to initialize the vector space.

    ##### outPorts:
    - memory: The Memory to set on AgentInit

    """

    api_key: InArg[secret]
    vector_space: InCompArg[str]
    initialize: InCompArg[bool]

    memory: OutArg[Memory]

    def execute(self, ctx) -> None:
        from vecto import Vecto

        api_key = os.getenv("VECTO_API_KEY") if self.api_key.value is None else self.api_key.value

        headers = {'Authorization': 'Bearer ' + api_key}
        response = requests.get("https://api.vecto.ai/api/v0/account/space", headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get vector space list: {response.text}")
        for space in response.json():
            if space['name'] == self.vector_space.value:
                vs = Vecto(api_key, space['id'])
                if self.initialize.value:
                    vs.delete_vector_space_entries()
                self.memory.value = VectoMemoryImpl(vs)
                break
        if not self.memory.value:
            raise Exception(f"Could not find vector space with name {self.vector_space.value}")


# TBD
#@xai_component
#class AgentToolbeltFolder(Component):
#    folder: InCompArg[str]
#
#    toolbelt_spec: OutArg[list]
#
#    def execute(self, ctx) -> None:
#        spec = []
#        self.toolbelt_spec.value = spec


@xai_component
class AgentInit(Component):
    """Initialize the agent with the necessary components.

    ##### inPorts:
    - agent_name: The name of the agent to create.
    - agent_provider: The provider of the agent (Either openai or vertexai).
    - agent_model: The model that the agent should use (Such as gpt-3.5-turbo, or gemini-pro).
    - agent_memory: The memory that the agent should use to store data it wants to remember.
    - system_prompt: The system prompt of the agent be sure to speficy 
      {tool_instruction} and {tools} to explain how to use them.
    - max_thoughts: The maximum number of thoughts/tools the agent can use before it must respond to the user.
    - toolbelt_spec: The toolbelt the agent has access to.
    """

    agent_name: InCompArg[str]
    agent_provider: InCompArg[str]
    agent_model: InCompArg[str]
    agent_memory: InCompArg[Memory]
    system_prompt: InCompArg[str]
    max_thoughts: InArg[int]
    toolbelt_spec: InCompArg[dict]
    
    def execute(self, ctx) -> None:
        if self.agent_provider.value != 'openai' and self.agent_provider.value != 'vertexai':
            raise Exception(f"agent provider: {self.agent_provider.value} is not supported in this version of xai_agent.")

        ctx['agent_' + self.agent_name.value] = {
            'agent_toolbelt': self.toolbelt_spec.value,
            'agent_provider': self.agent_provider.value,
            'agent_memory': self.agent_memory.value,
            'agent_model': self.agent_model.value,
            'agent_system_prompt': self.system_prompt.value,
            'max_thoughts': self.max_thoughts.value
        }


def make_tools_prompt(toolbelt: dict) -> dict:
    ret = ''
    
    for key, value in toolbelt.items():
        ret += f'{key}: {value.description} (Inputs: {value.inputs}) (Outputs: {value.outputs})\n'

    recall = 'recall: Fuzzily looks up a previously remembered JSON memo in your memory. (Inputs: ["text"]) (Outputs: ["text"])\n'
    remember = 'remember: Remembers a prompt, and a json note pair for the future. (Inputs: ["text", "text"]) (Outputs: ["text"]) (Example: TOOL: remember "todo later" "{ \"tasks\": [\"buy milk\",\"take out the trash\"]}"\n'
    
    return { 
        'tools': ret,
        'recall': recall,
        'remember': remember,
        'memory': recall + remember,
        'tool_instruction': 'To use a tool write TOOL: in one line followed by the tool name and arguments, system will respond with the results.'
    }

def conversation_to_vertexai(conversation) -> str:
    ret = ""
    
    for message in conversation:
        ret += message['role'] + ":" + message['content']
        ret += "\n\n"
    
    return ret
 
@xai_component
class AgentRun(Component):
    """Run the agent with the given conversation.

    ##### branches:
    - on_thought: Called whenever the agent uses a tool.

    ##### inPorts:
    - agent_name: The name of the agent to run.
    - conversation: The conversation to send to the agent.

    ##### outPorts:
    - out_conversation: The conversation with the agent's responses.
    - last_response: The last response of the agent.

    """
    on_thought: BaseComponent

    agent_name: InCompArg[str]
    conversation: InCompArg[any]

    out_conversation: OutArg[list]
    last_response: OutArg[str]

    def execute(self, ctx) -> None:
        agent = ctx['agent_' + self.agent_name.value]

        model_name = agent['agent_model']
        toolbelt = agent['agent_toolbelt']
        system_prompt = agent['agent_system_prompt']

        # deep to avoid messing with the original system prompt.
        conversation = copy.deepcopy(self.conversation.value)

        if conversation[0]['role'] != 'system':
            conversation.insert(0, {'role': 'system', 'content': system_prompt.format(**make_tools_prompt(toolbelt))})
        else:
            conversation[0]['content'] = system_prompt.format(**make_tools_prompt(toolbelt))

        thoughts = 0
        stress_level = 0.0  # Raise temperature if there are failures.

        while thoughts < agent['max_thoughts']:
            thoughts += 1

            if thoughts == agent['max_thoughts']:
                conversation.append({"role": "system", "content": "Maximum tool usage reached. Tools Unavailable"})

            if agent['agent_provider'] == 'vertexai':
                response = self.run_vertexai(model_name, conversation, stress_level)
            elif agent['agent_provider'] == 'openai':
                response = self.run_openai(model_name, conversation, stress_level)
            else:
                raise ValueError("Unknown agent provider")

            conversation.append(response)

            if thoughts <= agent['max_thoughts'] and 'TOOL:' in response['content']:
                stress_level = self.handle_tool_use(ctx, agent, conversation, response['content'], toolbelt, stress_level)
            else:
                # Allow only one tool per thought.
                break

        self.out_conversation.value = conversation
        self.last_response.value = conversation[-1]['content']

    def run_vertexai(self, model_name, conversation, stress_level):
        inputs = conversation_to_vertexai(conversation)
        model = GenerativeModel(model_name)
        result = model.generate_content(
            inputs,
            generation_config={
                "max_output_tokens": 2048,
                "stop_sequences": [
                    "\n\nsystem:",
                    "\n\nuser:",
                    "\n\nassistant:"
                ],
                "temperature": stress_level + 0.5,
                "top_p": 1
            },
            safety_settings=[],
            stream=False,
        )

        if "assistant:" in result.text:
            response = {"role": "assistant", "content": result.text.split("assistant:")[-1]}
        else:
            response = {"role": "assistant", "content": result.text}
        return response

    def run_openai(self, model_name, conversation, stress_level):
        result = openai.chat.completions.create(
            model=model_name,
            messages=conversation,
            max_tokens=2000,
            temperature=stress_level
        )
        response = result.choices[0].message
        return {"role": "assistant", "content": response.content}

    def handle_tool_use(self, ctx, agent, conversation, content, toolbelt, stress_level):
        SubGraphExecutor(self.on_thought).do(ctx)
        
        lines = content.split("\n")
        for line in lines:
            if line.startswith("TOOL:"):
                command = line.split(":", 1)[1].strip()
                try:
                    tool_name = command.split(" ", 1)[0].strip()
                    tool_args = command.split(" ", 1)[1].strip()
                except Exception:
                    tool_name = command.strip()
                    tool_args = ""

                if tool_name == 'recall':
                    memory = agent['agent_memory']
                    tool_result = str(memory.query(tool_args, 3))
                    print(f"recall got result: {tool_result}", flush=True)
                    conversation.append({"role": "system", "content": tool_result})
                elif tool_name == 'remember':
                    self.remember_tool(agent, tool_args, conversation)
                else:
                    stress_level = self.run_tool(toolbelt, tool_name, tool_args, conversation, stress_level)
        return stress_level

    def remember_tool(self, agent, tool_args, conversation):
        memory = agent['agent_memory']
        prompt_start = tool_args.find('"')
        prompt_end = tool_args.find('"', prompt_start)
        prompt = tool_args[prompt_start + 1:prompt_end].strip()
        memo_start = tool_args.find('"', prompt_end)
        memo = tool_args[memo_start + 1:len(tool_args) - 1].replace('\"', '"')

        try:
            json_memo = json.loads(memo)
        except Exception:
            # Invalid JSON, so just store as a string.
            json_memo = '"' + memo + '"'

        memory.add('', prompt, json_memo)
        print(f"Added {prompt}: {memo} to memory", flush=True)
        conversation.append({"role": "system", "content": f"Memory {prompt} stored."})

    def run_tool(self, toolbelt, tool_name, tool_args, conversation, stress_level):
        try:
            tool_result = toolbelt[tool_name](tool_args)
            print(f"tool {tool_name} got result:")
            print(tool_result)
            conversation.append({"role": "system", "content": tool_result})
            return stress_level
        except KeyError as e:
            print(f"tool {tool_name} not found.")
            conversation.append({"role": "system", "content": "ERROR: Tool not available: " + str(e)})
        except Exception as e:
            print(f"tool {tool_name} got exception:")
            print(e)
            conversation.append({"role": "system", "content": "ERROR: " + str(e)})
        return min(stress_level + 0.1, 1.5)

def word_or_pair_generator(input_string):
    words = input_string.split(' ')

    for word in words:
        if len(word) > 10:
            for i in range(0, len(word), 2):
                yield word[i:i+2]
        else:
            yield word

        if word != words[-1]:
            yield ' '


@xai_component
class AgentStreamStringResponse(Component):
    """Creates a Stream response from a string.

    When using Converse it is better for the user to see the response word by word
    as if it was being typed out, like it is in ChatGPT.

    Use with the ConverseStreamRespond or ConverseStreamPartialResponse 
    component when using Converse.

    ##### inPorts:
    - input_str: The string to stream.

    ##### outPorts:
    - result_stream: The result of the string to stream.
    """
    
    input_str: InCompArg[str]
    
    result_stream: OutArg[any]
    
    def execute(self, ctx) -> None:
        self.result_stream.value = word_or_pair_generator(self.input_str.value)

