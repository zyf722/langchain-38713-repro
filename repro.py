from __future__ import annotations

import json

from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

SAMPLE_INPUT = {"foo": "hello", "bar": "world"}


def tool_call_args_from_schema(schema: dict) -> dict[str, str | dict[str, str]]:
    params = schema.get("function", {}).get("parameters", {})
    properties = params.get("properties", {}) if isinstance(params, dict) else {}
    if (
        isinstance(properties, dict)
        and "root" in properties
        and "foo" not in properties
    ):
        return {"root": SAMPLE_INPUT}
    return SAMPLE_INPUT


class InputState(TypedDict):
    foo: str
    bar: str


class FullState(InputState):
    extra: int


def node(state: FullState) -> FullState:
    return state


builder = StateGraph(FullState, input_schema=InputState)
builder.add_node("n", node)
builder.add_edge(START, "n")
builder.add_edge("n", END)
compiled = builder.compile(name="my_tool")

tool = compiled.as_tool(name="my_tool", description="Example tool.")
oai_schema = convert_to_openai_tool(tool)

print("=== args_schema class MRO ===")
assert tool.args_schema is not None and not isinstance(tool.args_schema, dict)
print([c.__name__ for c in tool.args_schema.__mro__])
print()

print("=== OpenAI tool schema, i.e. what the LLM sees ===")
print(json.dumps(oai_schema, indent=2))
print()

print("=== Simulating what the LLM sends back per the schema above ===")
llm_reply = tool_call_args_from_schema(oai_schema)
print(json.dumps(llm_reply, indent=2))
print()

print("=== tool.invoke(...) with that payload ===")
result = tool.invoke(llm_reply)
print("OK:", result)
