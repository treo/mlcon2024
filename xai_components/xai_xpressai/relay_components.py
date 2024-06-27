from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, secret, xai_component
import openai
import os
import requests
import shutil


class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        for message in self.conversation_history:
            print(f"{message['role']}: {message['content']}\n\n")

@xai_component
class XpressAIMakeConversation(Component):
    """Creates a conversation object to hold conversation history.
    """

    prev: InArg[Conversation]
    system_msg: InArg[str]
    user_msg: InArg[str]
    assistant_msg: InArg[str]
    function_msg: InArg[str]

    conversation: OutArg[Conversation]

    def execute(self, ctx) -> None:
        conv = Conversation()
        if self.prev.value is not None:
            conv.conversation_history.extend(self.prev.value.conversation_history)
        if self.system_msg.value is not None:
            conv.add_message("system", self.system_msg.value)
        if self.user_msg.value is not None:
            conv.add_message("user", self.user_msg.value)
        if self.assistant_msg.value is not None:
            conv.add_message("assistant", self.assistant_msg.value)
        if self.function_msg.value is not None:
            conv.add_message("function", self.function_msg.value)

        self.conversation.value = conv

@xai_component
class XpressAIAuthorize(Component):
    """Setup for the XpressAI Gateway client.
    """

    def execute(self, ctx) -> None:    
        openai.base_url = "https://relay.public.cloud.xpress.ai/v1/"
        openai.api_key =  os.getenv("XPRESSAI_API_TOKEN")
        ctx["openai_api_key"] = openai.api_key


@xai_component
class XpressAIChat(Component):
    """Interacts with a specified model in a conversation.

    ##### inPorts:
    - model_name: Name of the model to be used for conversation.
    - system_prompt: Initial system message to start the conversation.
    - user_prompt: Initial user message to continue the conversation.
    - conversation: A list of conversation messages. Each message is a dictionary with a "role" and "content".
    - max_tokens: The maximum length of the generated text.
    - temperature: Controls randomness of the output text.
    - count: Number of responses to generate.

    ##### outPorts:
    - completion: The generated text of the model's response.
    - out_conversation: The complete conversation including the model's response.
    """
    model_name: InCompArg[str]
    system_prompt: InArg[str]
    user_prompt: InArg[str]
    conversation: InArg[list]
    max_tokens: InArg[int]
    temperature: InArg[float]
    count: InArg[int]
    completion: OutArg[str]
    out_conversation: OutArg[list]
        
    def __init__(self):
        super().__init__()
        
    def execute(self, ctx) -> None:
        if self.conversation.value is not None:
            messages = self.conversation.value
        else:
            messages = []
        
        if self.system_prompt.value is not None:            
            messages.append({"role": "system", "content": self.system_prompt.value})
        if self.user_prompt.value is not None:
            messages.append({"role": "user", "content": self.user_prompt.value })
        
        if not messages:
            raise Exception("At least one prompt is required")
        
        print("sending")
        for message in messages:
            print(message)
        
        
        result = openai.chat.completions.create(
            model=self.model_name.value,
            messages=messages,
            max_tokens=self.max_tokens.value if self.max_tokens.value is not None else 128,
            temperature=self.temperature.value if self.temperature.value is not None else 1,
            n=self.count.value if self.count.value is not None else 1
        )
        

        if self.count.value is None or self.count.value == 1:
            response = result.choices[0].message
            messages.append({"role": "assistant", "content": response.content})
        self.completion.value = result.choices[0].message.content
        self.out_conversation.value = messages


@xai_component
class XpressAIStreamChat(Component):
    """Interacts with a specified model from OpenAI in a conversation, streams the response.

    #### Reference:
    - [OpenAI API](https://platform.openai.com/docs/api-reference/completions/create)

    ##### inPorts:
    - model_name: Name of the model to be used for conversation.
    - system_prompt: Initial system message to start the conversation.
    - user_prompt: Initial user message to continue the conversation.
    - conversation: A list of conversation messages. Each message is a dictionary with a "role" and "content".
    - max_tokens: The maximum length of the generated text.
    - temperature: Controls randomness of the output text.
    - count: Number of responses to generate.

    ##### outPorts:
    - completion: The generated text of the model's response.
    - out_conversation: The complete conversation including the model's response.
    """
    model_name: InCompArg[str]
    system_prompt: InArg[str]
    user_prompt: InArg[str]
    conversation: InArg[list]
    max_tokens: InArg[int]
    temperature: InArg[float]
    completion_stream: OutArg[any]
    
    
    def execute(self, ctx) -> None:
        if self.conversation.value is not None:
            messages = self.conversation.value
        else:
            messages = []
        
        if self.system_prompt.value is not None:            
            messages.append({"role": "system", "content": self.system_prompt.value})
        if self.user_prompt.value is not None:
            messages.append({"role": "user", "content": self.user_prompt.value })
        
        if not messages:
            raise Exception("At least one prompt is required")
        
        print("sending")
        
        for message in messages:
            print(message)
        
        
        result = openai.chat.completions.create(
            model=self.model_name.value,
            messages=messages,
            max_tokens=self.max_tokens.value if self.max_tokens.value is not None else 128,
            temperature=self.temperature.value if self.temperature.value is not None else 1,
            stream=True
        )
        
        def extract_text():
            for chunk in result:
                yield chunk.choices[0].delta.content

        self.completion_stream.value = extract_text()


@xai_component
class ForEachStreaming(Component):
    """Executes the on_item branch for every single item in the given collection

    #### inPorts:
    - list: A collection of items
    """
    list: InCompArg[list]
    
    on_item: BaseComponent
    item: OutArg[any]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finished = False
        self.iterable = None
        self.cur_step = None
    
    def execute(self, ctx) -> None:
        if self.iterable is None:
            self.iterable = iter(self.list.value)

        if self.cur_step is None:
            try:
                self.item.value = next(self.iterable)        
            except StopIteration as e:
                self.finished = True
                self.iterable = None
                self.cur_step = None
                return
            self.cur_step = self.on_item
        self.cur_step = self.cur_step.do(ctx)
            

    def do(self, ctx) -> BaseComponent:
        self.execute(ctx)
        if self.finished:
            self.finished = False
            return self.next
        else:
            return self
        


@xai_component
class TakeNthElement(Component):
    """Takes the nth element from a list.

    ##### inPorts:
    - values: List from which the nth element will be taken.
    - index: Index of the element to take.

    ##### outPorts:
    - out: The nth element of the list.
    """

    values: InCompArg[list]
    index: InCompArg[int]
    out: OutArg[any]
    
    def execute(self, ctx) -> None:
        self.out.value = self.values.value[self.index.value]


@xai_component
class FormatConversation(Component):
    """Formats a conversation by appending messages to it.

    ##### inPorts:
    - prev_conversation: List of previous conversation messages.
    - system_prompt: Message to be appended from the system.
    - user_prompt: Message to be appended from the user.
    - faux_assistant_prompt: Message to be appended from the assistant.
    - input_prompt: Message to be appended from the user or system.
    - input_is_system: Boolean indicating whether the input prompt is from the system.
    - args: Arguments for formatting the messages.

    ##### outPorts:
    - out_messages: The formatted conversation.
    """
    prev_conversation: InArg[list]
    system_prompt: InArg[str]
    user_prompt: InArg[str]
    faux_assistant_prompt: InArg[str]
    input_prompt: InArg[str]
    input_is_system: InArg[bool]
    args: InArg[dict]
    out_messages: OutArg[list]
    
    def execute(self, ctx) -> None:
        conversation = [] if self.prev_conversation.value is None else self.prev_conversation.value
        format_args = {} if self.args.value is None else self.args.value
        
        
        if self.system_prompt.value is not None:
            conversation.append(self.make_msg('system', self.system_prompt.value.format(**format_args)))
            
        if self.user_prompt.value is not None:
            conversation.append(self.make_msg('user', self.user_prompt.value.format(**format_args)))
        
        if self.faux_assistant_prompt.value is not None:
            conversation.append(self.make_msg('assistant', self.faux_assistant_prompt.value.format(**format_args)))

        if self.input_prompt.value is not None:
            conversation.append(self.make_msg('system' if self.input_is_system.value else 'user', self.input_prompt.value.format(**format_args)))
        
        self.out_messages.value = conversation
        
        
    def make_msg(self, role, msg) -> dict:
        return { 'role': role, 'content': msg }

@xai_component
class AppendConversationResponse(Component):
    """Appends a response from the assistant to a conversation.

    ##### inPorts:
    - conversation: List of current conversation messages.
    - assistant_message: Message to be appended from the assistant.

    ##### outPorts:
    - out_conversation: The conversation including the assistant's response,
        ie: conversation + [{ 'role': 'assistant', 'content': assistant_message }]
    """
    conversation: InCompArg[list]
    assistant_message: InCompArg[str]
    out_conversation: OutArg[list]
    
    def execute(self, ctx) -> None:
        ret = self.conversation.value + [{ 'role': 'assistant', 'content': self.assistant_message.value}]
        self.out_conversation.value = ret

        
@xai_component
class JoinConversations(Component):
    """Appends multiple conversation lists into a single list.

    ##### inPorts:
    - conversation_1: First conversation to join.
    - conversation_2: Second conversation to join.
    - conversation_3: Third conversation to join.

    ##### outPorts:
    - out_conversation: The joined conversation.
    """

    conversation_1: InArg[list]
    conversation_2: InArg[list]
    conversation_3: InArg[list]
    
    out_conversation: OutArg[list]
    
    def execute(self, ctx) -> None:
        ret = []
        
        if self.conversation_1.value is not None:
            ret = ret + self.conversation_1.value
        if self.conversation_2.value is not None:
            ret = ret + self.conversation_2.value
        if self.conversation_3.value is not None:
            ret = ret + self.conversation_3.value
            
        self.out_conversation.value = ret

