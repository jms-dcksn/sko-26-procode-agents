from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel
from langchain_core.messages import AIMessage, ToolMessage


class AgentStreamHandler(BaseModel):
    """Utility to stream an agent and print a detailed trace of state changes."""

    step: int = 0

    def stream(self, agent: Any, messages: Any) -> None:
        """Stream from the agent and print a detailed trace for each chunk.

        Example usage in your notebook:

            from utils import AgentStreamHandler

            handler = AgentStreamHandler()
            handler.stream(agent, messages)
        """

        for chunk in agent.stream({"messages": messages}, stream_mode="values"):
            self._handle_chunk(chunk)

    def _handle_chunk(self, chunk: Dict[str, Any]) -> None:
        """Pretty-print a single chunk from the agent stream."""
        self.step += 1
        state = chunk
        latest_message = state["messages"][-1]

        print("\n" + "=" * 80)
        print(f"STEP {self.step}  |  time={datetime.now().isoformat(timespec='seconds')}")
        print(f"State keys: {list(state.keys())}")
        print("-" * 80)

        # Basic info about the latest message
        role = getattr(latest_message, "type", getattr(latest_message, "role", "unknown"))
        print(f"Latest message role: {role}")
        print(f"Latest message class: {latest_message.__class__.__name__}")

        # Categorize the event
        if isinstance(latest_message, AIMessage) and getattr(latest_message, "tool_calls", None):
            print("[AI → TOOL CALL]")
            for i, tc in enumerate(latest_message.tool_calls):
                print(f"  Tool call {i}:")
                print(f"    name: {tc.get('name')}")
                print(f"    args: {tc.get('args')}")
        elif isinstance(latest_message, ToolMessage):
            print("[TOOL → RESULT]")
            print(f"  tool name: {getattr(latest_message, 'name', 'unknown')}")
            print("  result content:")
            print(f"    {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            # This is typically an AI response without a tool call
            print("[AI → RESPONSE]")
            print("  content:")
            print(f"    {latest_message.content}")
        else:
            # e.g. HumanMessage or anything else
            print("[OTHER MESSAGE]")
            print("  content:")
            print(f"    {latest_message.content}")

        print("=" * 80)