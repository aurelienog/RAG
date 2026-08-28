# --------------------------
# DEFAULT
# --------------------------

all: install

# --------------------------
# ARGUMENTS FOR RUN/DEBUG
# --------------------------
# If the first argument is "run" or "debug", extract all subsequent words
# and treat them as arbitrary arguments for the command line interface.
ifeq ($(firstword $(MAKECMDGOALS)),$(filter $(firstword $(MAKECMDGOALS)),run debug))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # Turn the arguments into target placeholders so Make doesn't throw errors
  $(eval $(RUN_ARGS):;@:)
endif

EOF :=
ifndef RUN_ARGS
  RUN_ARGS := index
endif

# --------------------------
# INSTALL
# --------------------------

install:
	uv sync

# --------------------------
# RUN
# --------------------------

run:
	uv run python -m src $(RUN_ARGS)

debug:
	uv run python -m pdb -m src $(RUN_ARGS)

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
	uv run flake8 src
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy . --strict

# --------------------------
# PHONY
# --------------------------

.PHONY: all install run debug fclean clean lint lint-strict
