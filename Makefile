export OPENAI_KEY := $(shell jq -r '.openai_key' secret_key.json)
#  ssh -i ~/.ssh/id_ed25519 mischa@136.112.23.201

install:
	@echo "--- 🚀 Installing project dependencies ---"
	python3 -m venv .gym && . .gym/bin/activate && \
	pip install -e ./browsergym/core -e ./browsergym/miniwob -e ./browsergym/webarena -e ./browsergym/webarenalite -e ./browsergym/visualwebarena/ -e ./browsergym/experiments -e ./browsergym/assistantbench -e ./browsergym/ openai
	playwright install chromium

install-demo:
	@echo "--- 🚀 Installing demo dependencies ---"
	pip install -r demo_agent/requirements.txt
	playwright install chromium

# brew install wget first
install-agentbeats:
	pip install git+https://github.com/agentbeats/agentbeats.git@main && pip install openai
	@echo "Environment set up for AgentBeats."
	wget -O red_agent_card.toml https://raw.githubusercontent.com/agentbeats/agentbeats/main/scenarios/templates/template_tensortrust_red_agent/red_agent_card.toml && \
	sed -i '' 's/agent_url = "http:\/\/127.0.0.1:8000"/agent_url = "http:\/\/136.112.23.201:8000"/' red_agent_card.toml && \
	sed -i '' 's/launcher_url = "http:\/\/127.0.0.1:8080"/launcher_url = "http:\/\/136.112.23.201:8080"/' red_agent_card.toml && \
	sed -i '' 's/name = "TensorTrust"/name = "Benchwarmer Agent"/' red_agent_card.toml
	@echo "Agent card downloaded and modified with our IP/port."
	. .gym/bin/activate && agentbeats run red_agent_card.toml \
		--launcher_host 136.112.23.201 \
		--launcher_port 8080 \
		--agent_host 136.112.23.201 \
		--agent_port 8000
	@echo "Running AgentBeats currently"

install-test:
	pip install pytest pytest-xdist

setup-miniwob:
	@echo "--- 🤖 Setting up MiniWoB++ ---"
	@if [ ! -d "miniwob-plusplus" ]; then \
		echo "Cloning MiniWoB++ repository..."; \
		git clone https://github.com/Farama-Foundation/miniwob-plusplus.git; \
	else \
		echo "MiniWoB++ repository already exists, skipping clone..."; \
	fi
	@echo "Resetting to specific commit for reproducibility..."
	git -C "./miniwob-plusplus" reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
	@echo "Adding MINIWOB_URL to .env file..."
	@echo "MINIWOB_URL=\"file://$(shell pwd)/miniwob-plusplus/miniwob/html/miniwob/\"" >> .env
	@echo "✅ MiniWoB++ setup complete!"
	@echo "💡 To use MiniWoB++, load the environment variables:"
	@echo "   source .env"

setup-agentbeat-battle:
	curl -X POST https://agentbeats.org/battle/register \
		-d "agent_url=http://136.112.23.201:8000" \
		-d "battle_name=example_battle" && \
	@echo "Battle registered."

demo:
	@echo "--- 🚀 Running open ended demo agent ---"
	(set -x && cd demo_agent && python run_demo.py)

# ex: make demo-tasks TASKS="miniwob.click-dialog miniwob.click-test miniwob.choose-date"
demo-tasks:
	@echo "--- 🚀 Running demo agent with tasks ---"
	@cd demo_agent && \
	for TASK in $(TASKS); do \
		BENCHMARK=$$(echo $$TASK | cut -d. -f1); \
		[ "$$BENCHMARK" = "miniwob" ] && export MINIWOB_URL="file://$(shell pwd)/miniwob-plusplus/miniwob/html/miniwob/"; \
		[ "$$BENCHMARK" = "webarena" ] && export WEB_ARENA_DATA_DIR="$(shell pwd)/../bg_wl_data"; \
		[ "$$BENCHMARK" = "visualwebarena" ] && export DATASET=visualwebarena && export VWA_CLASSIFIEDS="http://localhost:9980" && export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c" && \
			export VWA_SHOPPING="http://localhost:7770" && export VWA_REDDIT="http://localhost:9999" && export VWA_WIKIPEDIA="http://localhost:8888" && export VWA_HOMEPAGE="http://localhost:4399"; \
		[ "$$BENCHMARK" = "workarena" ] && export WORK_ARENA_INSTANCE_URL="$(shell pwd)/../workarena_env"; \
		[ "$$BENCHMARK" = "weblinx" ] && export WEBLINX_DATA_DIR="$(shell pwd)/../bg_wl_data"; \
		python run_demo.py --task $$TASK; \
	done

test-core:
	@echo "--- 🧪 Running tests ---"
	pytest -n auto ./tests/core

clean-miniwob:
	@echo "--- 🧹 Cleaning MiniWoB++ installation ---"
	rm -rf miniwob-plusplus
	@echo "✅ MiniWoB++ installation cleaned!"

help:
	@echo "Available targets:"
	@echo "  install          - Install project dependencies"
	@echo "  setup-miniwob    - Setup MiniWoB++ dependencies"
	@echo "  install-demo     - Install demo dependencies"
	@echo "  demo             - Run demo agent"
	@echo "  test-core        - Run core tests"
	@echo "  clean-miniwob    - Remove MiniWoB++ directory"
	@echo "  help             - Show this help message"

.PHONY: install setup-miniwob install-demo demo test-core clean-miniwob help
