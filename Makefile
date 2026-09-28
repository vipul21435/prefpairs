.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format typecheck test cov check demo docker docker-run docker-demo clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Create the virtualenv from uv.lock and install git hooks
	$(UV) sync --frozen
	$(UV) run pre-commit install

lint: ## Ruff lint and format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format: ## Apply ruff fixes and formatting
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

typecheck: ## mypy --strict over src/
	$(UV) run mypy

test: ## Run the test suite
	$(UV) run pytest -q

cov: ## Run tests with branch coverage (fails under 85%)
	$(UV) run pytest -q --cov --cov-report=term-missing --cov-report=xml

check: lint typecheck cov ## Everything CI runs

DEMO_DB ?= .prefpairs/demo.db

demo: ## End to end on the bundled sample: import, stats, rank with CIs, check recovery
	PREFPAIRS="$(UV) run prefpairs" DEMO_DB=$(DEMO_DB) sh scripts/demo.sh

IMAGE ?= prefpairs:local

docker: ## Build the CLI image (label project=prefpairs) and prune dangling layers
	docker build -t $(IMAGE) .
	docker image prune -f --filter label=project=prefpairs

docker-run: ## Run the CLI in the image, e.g. make docker-run ARGS="simulate --seed 0"
	docker run --rm $(IMAGE) $(ARGS)

docker-demo: docker ## Run the end-to-end demo inside the image
	docker run --rm --entrypoint sh $(IMAGE) scripts/demo.sh

clean: ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
