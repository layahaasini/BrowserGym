.SILENT:

include .env
export $(shell sed 's/=.*//' .env)

PY := $(shell pwd)/.gym/bin/python
PIP := $(shell pwd)/.gym/bin/pip

install:
	@echo "--- 🚀 Configuring environment and installing dependencies ---"
	@if [ "$$(uname)" = "Linux" ]; then \
		sudo apt-get update && \
		sudo apt-get install -y xvfb libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libatspi2.0-0 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2 docker.io python3-pip && \
		sudo pip3 install docker-compose && sudo usermod -aG docker $USER; \
	else \
		echo "Skipping apt-get (not Linux)."; \
	fi
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
	sed -i 's/agent_url = "http:\/\/127.0.0.1:8000"/agent_url = "http:\/\/$(AGENT_HOST):$(AGENT_PORT)"/' red_agent_card.toml
	sed -i 's/launcher_url = "http:\/\/127.0.0.1:8080"/launcher_url = "http:\/\/$(LAUNCHER_HOST):$(LAUNCHER_PORT)"/' red_agent_card.toml
	sed -i 's/name = "TensorTrust"/name = "$(AGENT_NAME)"/' red_agent_card.toml
	$(PY) -m agentbeats run red_agent_card.toml \
		--launcher_host $(LAUNCHER_HOST) \
		--launcher_port $(LAUNCHER_PORT) \
		--agent_host $(AGENT_HOST) \
		--agent_port $(AGENT_PORT)
	@echo "Running AgentBeats currently"

register-battle:
	curl -X POST https://agentbeats.org/battle/register \
		-d "agent_url=http://$(AGENT_HOST):$(AGENT_PORT)" \
		-d "battle_name=$(BATTLE_NAME)"
	@echo "Battle registered."

demo:
	@echo "--- Running demo agent with tasks ---"
	cd demo_agent && \
		export $$(grep -v '^#' ../.env | xargs); \
		if [ -z "$(TASKS)" ]; then \
			echo "No TASKS specified — running open-ended demo agent"; \
			xvfb-run -a $(PY) run_demo.py; \
		else \
			for TASK in $(TASKS); do \
				echo "\n=== Working on task: $$TASK ==="; \
				BENCHMARK=$$(echo $$TASK | cut -d. -f1); \
				[ "$$BENCHMARK" = "webarena" ] && export WEB_ARENA_DATA_DIR="$$WEB_ARENA_DATA_DIR"; \
				[ "$$BENCHMARK" = "visualwebarena" ] && export DATASET="$$DATASET"; \
				xvfb-run -a $(PY) run_demo.py --task $$TASK; \
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
