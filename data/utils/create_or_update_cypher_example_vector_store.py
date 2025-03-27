"""
This script will create or update a Cypher query vector store in the specified Neo4j database.
The contents will be the YAML file specified.
Questions will be embedded using OpenAI's text-embedding-ada-002 model.
"""

import os
import sys

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings

from ps_genai_agents.ingest.cypher_examples import (
    embed_cypher_query_nodes,
    get_existing_questions,
    load_cypher_query_nodes,
    read_cypher_examples_from_yaml_file,
    remove_preexisting_nodes_from_ingest_tasks,
)

if load_dotenv():
    print("Successfully loaded environment from .env")
else:
    print("Unable to load environment from .env")


def main() -> None:
    print("\n-----")
    print(f"Running Script {sys.argv[0]}")
    assert len(sys.argv) > 1, "File path not detected in arguments."
    print(f"Reading Data From File: {sys.argv[1]}")
    file_path: str = sys.argv[1]

    driver: Driver = GraphDatabase.driver(
        uri=os.getenv("NEO4J_URI", ""),
        auth=(os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", "")),
    )

    MODEL_NAME = "text-embedding-ada-002"
    embedder = OpenAIEmbeddings(model=MODEL_NAME)

    # Create a vector index if it doesn't exist
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        session.run("""
CREATE VECTOR INDEX cypher_query_vector_index IF NOT EXISTS
FOR (m:CypherQuery)
ON m.questionEmbedding
OPTIONS {
    indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
                }
        }
""")
    # read question and cql pairs from a YAML file
    # this is optional as long as there is a list of Python dictionaries prepared for processing
    unembedded_tasks = read_cypher_examples_from_yaml_file(file_path=file_path)
    print(f"Found {len(unembedded_tasks)} tasks...")

    # we optionally get the preexisting node ids (question property) as a set
    # we can remove these from the tasks
    preexisting_nodes = get_existing_questions(driver)
    print(f"Found {len(preexisting_nodes)} questions in Neo4j database...")

    # remove any questions we already have in the database
    cleaned_tasks = remove_preexisting_nodes_from_ingest_tasks(
        unembedded_tasks, preexisting_nodes
    )
    if len(cleaned_tasks) > 0:
        print(f"Embedding {len(cleaned_tasks)} new questions...")

        # embed the remaining tasks
        # this embeds the question content
        embedded_tasks_results = embed_cypher_query_nodes(
            embedder, cleaned_tasks, MODEL_NAME
        )

        # access the prepared nodes with embeddings with the `nodes` key
        embedded_tasks = embedded_tasks_results.get("nodes", list())
        print(f"Successfully embedded {len(embedded_tasks)} questions!")

        # any tasks that generated errors during the embedding process can be accessed with the `failed` key
        failed_embeddings = embedded_tasks_results.get("failed")
        if failed_embeddings is not None:
            print(f"Failed to embed {len(failed_embeddings)} tasks:")
            [print(f"* {t}") for t in failed_embeddings]

        print(f"Loading {len(embedded_tasks)} nodes into Neo4j...")
        # load CypherQuery nodes into Neo4j
        load_cypher_query_nodes(driver, embedded_tasks)

        print("Ingest Complete!")
    else:
        print("No new tasks found.")
        print("Process Complete!")


if __name__ == "__main__":
    main()
