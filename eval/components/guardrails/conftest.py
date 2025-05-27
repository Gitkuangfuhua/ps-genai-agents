from typing import Any, Callable, Coroutine, Dict
from unittest.mock import MagicMock

import pytest
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

from ps_genai_agents.components.guardrails import create_guardrails_node
from ps_genai_agents.components.state import InputState


@pytest.fixture(scope="module")
def mock_neo4j_graph_bbc_recipes() -> MagicMock:
    g = MagicMock(spec=Neo4jGraph)

    g.get_schema.return_value = """Node properties:
- **Recipe**
  - `id`: STRING Example: "8096211"
  - `name`: STRING Example: "veggie rainbow picnic pie"
  - `description`: STRING Example: "packed with veggies, feta and egg, this vegetarian"
  - `skillLevel`: STRING Example: "easy"
  - `cookingTimeMinutes`: INTEGER Example: "100"
  - `preparationTimeMinutes`: INTEGER Example: "45"
- **Ingredient**
  - `name`: STRING Example: "'00' flour"
- **Keyword**
  - `name`: STRING Example: "pesto"
- **DietType**
  - `name`: STRING Example: "dairy-free"
- **Author**
  - `name`: STRING Example: ""
- **Collection**
  - `name`: STRING Example: "15 minutes or less"
- **CypherQuery**
  - `question`: STRING Example: "What are some chocolate cakes I could make?"
  - `cypherStatement`: STRING Example: "MATCH (:Collection {name: 'Chocolate cake'})<-[:CO"
  - `embeddingModel`: STRING Available options: ['text-embedding-ada-002']
Relationship properties:

The relationships:
(:Recipe)-[:CONTAINS_INGREDIENT]->(:Ingredient)
(:Recipe)-[:KEYWORD]->(:Keyword)
(:Recipe)-[:DIET_TYPE]->(:DietType)
(:Recipe)-[:COLLECTION]->(:Collection)
(:Author)-[:WROTE]->(:Recipe)"""

    return g


@pytest.fixture(scope="module")
def scope_description_bbc_recipes() -> str:
    return "This application may answer questions related to cooking recipes and their authors."


@pytest.fixture(scope="module")
def guardrails_node(
    llm_openai_gpt_4o: ChatOpenAI,
    mock_neo4j_graph_bbc_recipes: MagicMock,
    scope_description_bbc_recipes: str,
) -> Callable[[InputState], Coroutine[Any, Any, Dict[str, Any]]]:
    return create_guardrails_node(
        llm=llm_openai_gpt_4o,
        graph=mock_neo4j_graph_bbc_recipes,
        scope_description=scope_description_bbc_recipes,
    )


@pytest.fixture(scope="module")
def guardrails_langsmith_dataset_name() -> str:
    return "Guardrails Dataset - BBC Recipes"


@pytest.fixture(scope="function")
def validate_guardrails() -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    def _validate_guardrails(
        outputs: Dict[str, str], reference_outputs: Dict[str, str]
    ) -> bool:
        """Evaluate whether the guardrails node appropriately accepts the input as in scope."""

        return outputs["next_action"] == reference_outputs["answer"]

    return _validate_guardrails
