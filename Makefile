OPENAI_API_KEY := $(shell jq -r '.openai_api_key' secret_key.json)
export OPENAI_API_KEY

PY := .gym/bin/python
PIP := .gym/bin/pip

install:
	@echo "--- 🚀 Configuring environment and installing dependencies ---"
	python3 -m venv .gym
	@$(PIP) install --upgrade pip && $(PIP) install -r requirements.txt && $(PY) -m playwright install chromium
	@if [ ! -d "miniwob-plusplus" ]; then \
		git clone https://github.com/Farama-Foundation/miniwob-plusplus.git; \
	fi
	git -C miniwob-plusplus reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
	echo "MINIWOB_URL=\"file://$(shell pwd)/miniwob-plusplus/miniwob/html/miniwob/\"" >> .env
	@echo "✅ Environment setup complete."

install-agentbeats:
	@$(PIP) install git+https://github.com/agentbeats/agentbeats.git@main openai

register-agent:
	wget -O red_agent_card.toml https://raw.githubusercontent.com/agentbeats/agentbeats/main/scenarios/templates/template_tensortrust_red_agent/red_agent_card.toml
	sed -i 's/agent_url = "http:\/\/127.0.0.1:8000"/agent_url = "http:\/\/34.10.45.207:8000"/' red_agent_card.toml
	sed -i 's/launcher_url = "http:\/\/127.0.0.1:8080"/launcher_url = "http:\/\/34.10.45.207:8080"/' red_agent_card.toml
	sed -i 's/name = "TensorTrust"/name = "Benchwarmer Agent"/' red_agent_card.toml
	@echo "Agent card downloaded and modified with our IP/port."
	$(PY) -m agentbeats run red_agent_card.toml \
		--launcher_host 34.10.45.207 \
		--launcher_port 8080 \
		--agent_host 34.10.45.207 \
		--agent_port 8000
	@echo "Running AgentBeats currently"

register-battle:
	curl -X POST https://agentbeats.org/battle/register \
		-d "agent_url=http://34.10.45.207:8000" \
		-d "battle_name=test_battle"
	@echo "Battle registered."

demo:
	@echo "--- 🚀 Running demo agent with tasks ---"
	cd demo_agent && \
	if [ -f "../.env" ]; then \
		echo "Loading environment from ../.env"; \
		export $$(grep -v '^#' ../.env | xargs); \
	fi; \
	if [ -z "$(TASKS)" ]; then \
		echo "No TASKS specified — 🚀 running open-ended demo agent"; \
		$(PY) run_demo.py; \
	else \
		for TASK in $(TASKS); do \
			echo "\n=== 🧠 Working on task: $$TASK ==="; \
			BENCHMARK=$$(echo $$TASK | cut -d. -f1); \
			[ "$$BENCHMARK" = "miniwob" ] && export MINIWOB_URL="file://$(shell pwd)/miniwob-plusplus/miniwob/html/miniwob/"; \
			[ "$$BENCHMARK" = "webarena" ] && export WEB_ARENA_DATA_DIR="$(shell pwd)/../bg_wl_data"; \
			[ "$$BENCHMARK" = "visualwebarena" ] && export DATASET=visualwebarena && \
				export VWA_CLASSIFIEDS="http://localhost:9980" && \
				export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c" && \
				export VWA_SHOPPING="http://localhost:7770" && \
				export VWA_REDDIT="http://localhost:9999" && \
				export VWA_WIKIPEDIA="http://localhost:8888" && \
				export VWA_HOMEPAGE="http://localhost:4399"; \
			$(PY) run_demo.py --task $$TASK; \
			echo "\n✅ Finished task: $$TASK"; \
		done; \
	fi

test-core:
	@echo "--- 🧪 Running tests ---"
	$(PY) -m pytest -n auto ./tests/core

clean-miniwob:
	@echo "--- 🧹 Cleaning MiniWoB++ installation ---"
	rm -rf miniwob-plusplus
	@echo "✅ MiniWoB++ installation cleaned!"

help:
	@echo "Available targets:"
	@echo "  install               - Install project/demo/benchmark/test dependencies"
	@echo "  install-agentbeats    - Install agentbeats and register agent"
	@echo "  register-agent        - Register your agent onto the Agentbeats platform"
	@echo "  register-battle       - Register your agent for an Agentbeats battle"
	@echo "  demo                  - Run demo agent"
	@echo "  test-core             - Run core tests"
	@echo "  clean-miniwob         - Remove MiniWoB++ directory"
	@echo "  help                  - Show this help message"

.PHONY: install install-agentbeats register-agent register-battle demo test-core clean-miniwob help
