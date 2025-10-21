install:
	@echo "--- 🚀 Installing project dependencies ---"
	pip install -e ./browsergym/core -e ./browsergym/miniwob -e ./browsergym/webarena -e ./browsergym/webarenalite -e ./browsergym/visualwebarena/ -e ./browsergym/experiments -e ./browsergym/assistantbench -e ./browsergym/
	playwright install chromium

install-demo:
	@echo "--- 🚀 Installing demo dependencies ---"
	pip install -r demo_agent/requirements.txt
	playwright install chromium

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
