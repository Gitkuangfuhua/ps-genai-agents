"""
A tool_selection node must
* take a single task at a time
* retrieve a list of available tools
    * text2cypher
    * viz
        * may dynamically provide this according to Task details
    * custom pre-written cypher executors
        * these can be numerous and may be retrieved in the same fashion as CypherQuery node contents
    * unstructured text search (sim search)
* decide the appropriate tool for the task
* generate and validate parameters for the selected tool
* send the validated parameters to the appropriate tool node
"""

from typing import Any, Callable, Coroutine, Dict, List, Literal, Set

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables.base import Runnable
from langgraph.types import Command, Send
from pydantic import BaseModel

from ...components.state import ToolSelectionInputState
from ...components.tool_selection.prompts import create_tool_selection_prompt_template

tool_selection_prompt = create_tool_selection_prompt_template()


def create_tool_selection_node(
    llm: BaseChatModel,
    tool_schemas: List[type[BaseModel]],
    default_to_text2cypher: bool = True,
) -> Callable[[ToolSelectionInputState], Coroutine[Any, Any, Command[Any]]]:
    """
    Create a tool_selection node to be used in a LangGraph workflow.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM used to process data.
    tool_schemas : Sequence[Union[Dict[str, Any], type, Callable, BaseTool]
        tools schemas that inform the LLM which tools are available.
    default_to_text2cypher : bool, optional
        Whether to attempt Text2Cypher if no tool calls are returned by the LLM, by default True

    Returns
    -------
    Callable[[ToolSelectionInputState], ToolSelectionOutputState]
        The LangGraph node.
    """

    tool_selection_chain: Runnable[Dict[str, Any], Any] = (
        tool_selection_prompt
        | llm.bind_tools(tools=tool_schemas)
        | PydanticToolsParser(tools=tool_schemas, first_tool_only=True)
    )

    # get a set of tool names that require the custom cypher executor
    predefined_cypher_tools: Set[str] = {
        t.model_json_schema().get("title", "") for t in tool_schemas
    }

    predefined_cypher_tools.discard("text2cypher")
    predefined_cypher_tools.discard("visualize")

    async def tool_selection(
        state: ToolSelectionInputState,
    ) -> Command[Literal["text2cypher", "error_tool_selection", "predefined_cypher"]]:
        """
        Choose the appropriate tool for the given task.
        """

        go_to_text2cypher: Command[
            Literal[
                "text2cypher", "error_tool_selection", "predefined_cypher"
            ]  # obviously this only goes to text2cypher, but this definition satisfies mypy...
        ] = Command(
            goto=Send(
                "text2cypher",
                {
                    "task": state.get("question", ""),
                    "prev_steps": ["tool_selection"],
                },
            )
        )

        # if possible determine tool without LLM
        if (
            len(predefined_cypher_tools) == 1
            and predefined_cypher_tools.pop().lower() == "text2cypher"
        ):
            return go_to_text2cypher

        # use LLM to determine tool
        tool_selection_output: BaseModel = await tool_selection_chain.ainvoke(
            {"question": state.get("question", "")}
        )
        print(tool_selection_output)
        # route to chosen tool node
        if tool_selection_output is not None:
            tool_name: str = tool_selection_output.model_json_schema().get("title", "")
            tool_args: Dict[str, Any] = tool_selection_output.model_dump()

            if tool_name in predefined_cypher_tools:
                return Command(
                    goto=Send(
                        "predefined_cypher",
                        {
                            "task": state.get("question", ""),
                            "query_name": tool_name,
                            "query_parameters": tool_args,
                            "prev_steps": ["tool_selection"],
                        },
                    )
                )
            elif tool_name == "text2cypher":
                return go_to_text2cypher

        elif default_to_text2cypher:
            return go_to_text2cypher

        # handle instance where no tool is chosen
        else:
            return Command(
                goto=Send(
                    "error_tool_selection",
                    {
                        "task": state.get("question", ""),
                        "errors": [
                            f"Unable to assign tool to question: `{state.get('question', '')}`"
                        ],
                        "prev_steps": ["tool_selection"],
                    },
                )
            )

        return go_to_text2cypher

    return tool_selection
