# =============================================================================
# DDM501 Lab 4 — one entry point per thing you might want to do.
#
# Same principle as Lab 3: the command CI runs and the command you run are the
# same string, so "works on my machine" cannot quietly become true.
# =============================================================================
.PHONY: help install train reference test test-fast lint check ci clean \
        up down logs load drift unfair promtool dashboards open

PYTHON   ?= python
PYTEST   ?= $(PYTHON) -m pytest
COMPOSE  ?= docker compose
PROMTOOL ?= promtool

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:  ## Train the model artifact the service loads
	$(PYTHON) scripts/train_model.py

reference:  ## Freeze the training distribution that drift is measured against
	$(PYTHON) scripts/make_reference.py

# -----------------------------------------------------------------------------
test:  ## Run the whole suite with coverage
	$(PYTEST)

test-fast:  ## No coverage — the loop you run while editing
	$(PYTEST) -q --no-cov

promtool:  ## Validate and unit-test the Prometheus configuration
	$(PROMTOOL) check rules monitoring/prometheus/alerts/*.yml
	cd monitoring/prometheus/tests && $(PROMTOOL) test rules alert_tests.yml

check: test promtool  ## Everything CI runs

# -----------------------------------------------------------------------------
up:  ## Start api + prometheus + grafana + node-exporter
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  API         http://localhost:8000/docs"
	@echo "  Metrics     http://localhost:8000/metrics"
	@echo "  Prometheus  http://localhost:9090"
	@echo "  Grafana     http://localhost:3000   (admin / admin)"

down:  ## Stop the stack
	$(COMPOSE) down

down-clean:  ## Stop the stack and delete its volumes
	$(COMPOSE) down -v

logs:  ## Follow the API logs
	$(COMPOSE) logs -f api

ps:  ## What is running
	$(COMPOSE) ps

# -----------------------------------------------------------------------------
load:  ## Normal traffic — establishes what healthy looks like
	$(PYTHON) scripts/load_test.py --profile normal --requests 400

drift:  ## Drifted traffic — the population moves, the model does not
	$(PYTHON) scripts/load_test.py --profile drifted --requests 400

drift-mild:  ## A shift small enough that only the drift metric notices
	$(PYTHON) scripts/load_test.py --profile drifted --strength 0.05 --requests 400

unfair:  ## One group's applications made systematically riskier
	$(PYTHON) scripts/load_test.py --profile unfair --requests 400

watch:  ## Live monitoring state in the terminal
	watch -n 2 'curl -s localhost:8000/monitoring | python -m json.tool'

clean:  ## Remove caches and coverage output
	rm -rf .pytest_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
