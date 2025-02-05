"""
This code is based on content found in the LangGraph documentation: https://python.langchain.com/docs/tutorials/graph/#advanced-implementation-with-langgraph
"""

from typing import Any, Callable, Coroutine, Dict, List

from langchain_neo4j import Neo4jGraph

from ..state import CypherOutputState, CypherState


def create_text2cypher_execution_node(
    graph: Neo4jGraph,
) -> Callable[
    [CypherState], Coroutine[Any, Any, Dict[str, List[CypherOutputState] | List[str]]]
]:
    """
    Create a Text2Cypher execution node for a LangGraph workflow.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.

    Returns
    -------
    Callable[[CypherState], Dict[str, List[CypherOutputState] | List[str]]]
        The LangGraph node.
    """

    no_results = [
        {"error": "I couldn't find any relevant information in the database."}
    ]

    async def execute_cypher(
        state: CypherState,
    ) -> Dict[str, List[CypherOutputState] | List[str]]:
        """
        Executes the given Cypher statement.
        """
        records = graph.query(state.get("statement", ""))
        steps = state.get("steps", list())
        steps.append("execute_cypher")
        return {
            "cyphers": [
                CypherOutputState(
                    **{
                        "subquestion": state.get("subquestion", ""),
                        "statement": state.get("statement", ""),
                        "errors": state.get("errors", list()),
                        "records": records if records else no_results,
                        "steps": steps,
                    }
                )
            ],
            "steps": ["text2cypher"],
        }

    return execute_cypher
