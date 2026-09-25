# --------------------------
# DEFAULT
# --------------------------

all: install

# --------------------------
# ARGUMENTS
# --------------------------

K ?= 10
QUERY ?= What is vLLM?
HOST ?= 127.0.0.1
PORT ?= 8000

# --------------------------
# INSTALL
# --------------------------

install:
	uv sync

# --------------------------
# RUN
# --------------------------

run: index

debug:
	uv run python -m pdb -m src index

# --------------------------
# CMD
# --------------------------

index:
	uv run python -m src index

index-semantic:
	uv run python -m src index-semantic

search:
	uv run python -m src search --query="$(QUERY)" --k=$(K)

search-semantic:
	uv run python -m src search-semantic --query="$(QUERY)" --k=$(K)

search-hybrid:
	uv run python -m src search-hybrid --query="$(QUERY)" --k=$(K)

search-dataset:
	uv run python -m src search-dataset --k=$(K)

answer:
	uv run python -m src answer --query="$(QUERY)" --k=$(K)

answer-dataset:
	uv run python -m src answer-dataset

evaluate:
	uv run python -m src evaluate

api:
	uv run python -m src api \
		--host="$(HOST)" \
		--port=$(PORT)

# --------------------------
# CLEAN
# --------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# --------------------------
# FCLEAN
# --------------------------

fclean: clean
	rm -rf .venv
	@echo "💣 Virtual environment removed"

# --------------------------
# LINT
# --------------------------

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

# --------------------------
# PHONY
# --------------------------

.PHONY: all install run debug fclean clean lint lint-strict \
index search search-dataset answer answer-dataset evaluate  \
index-semantic search-semantic search-hybrid caching api
