import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ps_genai_agents.retrievers.cypher_examples import Neo4jVectorSearchCypherExampleRetriever
from ps_genai_agents.ui.components import chat, display_chat_history, sidebar
from ps_genai_agents.workflows.multi_agent import (
    create_multi_tool_workflow,
)
from data.bbc_recipes.queries import get_cypher_statements_dictionary, get_tool_schemas
from ps_genai_agents.components.text2cypher import get_text2cypher_schema

if load_dotenv():
    print("Env Loaded Successfully!")
else:
    print("Unable to Load Environment.")


def get_args() -> Dict[str, Any]:
    """Parse the command line arguments to configure the application."""

    args = sys.argv
    if len(args) > 1:
        config_path: str = args[1]
        assert config_path.lower().endswith(
            ".json"
        ), f"provided file is not JSON | {config_path}"
        with open(config_path, "r") as f:
            config: Dict[str, Any] = json.load(f)
    else:
        config = dict()

    return config


def initialize_state(
    scope_description: str,
    example_questions: List[str] = list(),
) -> None:
    """
    Initialize the application state.
    """

    if "agent" not in st.session_state:
        st.session_state["llm"] = ChatOpenAI(model="gpt-4o", temperature=0.0)
        st.session_state["graph"] = Neo4jGraph(
            url=os.environ.get("NEO4J_URI"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD"),
            enhanced_schema=True,
            driver_config={"liveness_check_timeout": 0},
        )

        st.session_state["embedder"] = OpenAIEmbeddings(model="text-embedding-ada-002")

        cypher_example_retriever = Neo4jVectorSearchCypherExampleRetriever(neo4j_database="neo4j", neo4j_driver=st.session_state["graph"]._driver, vector_index_name="cypher_query_vector_index", embedder=st.session_state["embedder"])

        cypher_queries_for_tools = (
            get_cypher_statements_dictionary()
        )  # this is used to find Cypher queries based on a name

        tool_schemas = (
            get_tool_schemas() + [get_text2cypher_schema()]
        )  # these are Pydantic classes that define the available Cypher queries and their parameters


        
        st.session_state["agent"] = create_multi_tool_workflow(
            llm=st.session_state["llm"],
            graph=st.session_state["graph"],
            tool_schemas=tool_schemas,
            predefined_cypher_dict=cypher_queries_for_tools,
            scope_description=scope_description,
            cypher_example_retriever=cypher_example_retriever,
            llm_cypher_validation=False,
            attempt_cypher_execution_on_final_attempt=True,
            default_to_text2cypher=True,
        )
        st.session_state["messages"] = list()
        st.session_state["example_questions"] = example_questions


async def run_app(title: str = "Neo4j GenAI Demo") -> None:
    """
    Run the Streamlit application.
    """

    st.title(title)
    sidebar()
    display_chat_history()
    # Prompt for user input and save and display
    if question := st.chat_input():
        st.session_state["current_question"] = question

    if "current_question" in st.session_state:
        await chat(str(st.session_state.get("current_question", "")))


if __name__ == "__main__":
    args = get_args()
    initialize_state(
        scope_description=args.get("scope_description", ""),
        example_questions=args.get("example_questions", list()),
    )
    asyncio.run(run_app(title=args.get("title", "")))
