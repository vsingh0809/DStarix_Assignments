from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The state of the graph.
    'messages' holds the conversation history and tool invocations.
    The 'add_messages' reducer ensures new messages are appended, not overwritten.
    """
    messages: Annotated[list, add_messages]