.PHONY: all format lint test tests free_tests integration_tests help

# Default target executed when no arguments are given to make.
all: help

######################
# SET UP
######################

init:
	poetry install --with dev, ui
	pre-commit install


######################
# TESTING
######################

test:
	poetry run pytest tests

test_local:
	docker compose -f tests/integration/docker-compose.yml up -d
	poetry run pytest tests -s
	docker compose -f tests/integration/docker-compose.yml stop

test_integration_local:
	docker compose -f tests/integration/docker-compose.yml up -d
	poetry run pytest tests/integration -s
	docker compose -f tests/integration/docker-compose.yml stop

test_integration:
	poetry run pytest tests/integration

test_unit:
	poetry run pytest tests/unit -s

######################
# LINTING AND FORMATTING
######################

format:
	poetry run ruff format
	poetry run ruff check --select I . --fix
	poetry run ruff check .

clean:
	poetry run ruff check --select I . --fix
	poetry run ruff check . --fix

######################
# MYPY CHECK
######################

mypy:
	poetry run mypy .

######################
# DATA LOADING
######################

make load_iqs:
	poetry run python3 data/iqs/ingest/ingest_iqs.py
	poetry run python3 -m data.utils.create_or_update_cypher_example_vector_store ./data/iqs/queries/queries.yml

make load_bbc_recipes:
	poetry run python3 data/bbc_recipes/ingest/ingest_bbc_recipes.py
	poetry run python3 -m data.utils.create_or_update_cypher_example_vector_store ./data/bbc_recipes/queries/queries.yml


######################
# STREAMLIT APP
######################

make streamlit:
	poetry run streamlit run streamlit_app.py $(file_path)

make streamlit_bbc_recipes:
	poetry run streamlit run bbc_recipes_app.py data/bbc_recipes/ui-config.json

######################
# HELP
######################

help:
	@echo '----'
	@echo 'init........................ - initialize the repo for development'
	@echo 'init_workshop............... - initialize the repo for the workshop'
	@echo 'format...................... - run code formatters'
	@echo 'test........................ - run all unit and integration tests'
	@echo 'test_unit................... - run all free unit tests'
	@echo 'test_integration............ - run all integration tests'
