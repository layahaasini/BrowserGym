.SILENT:

-include .env
export $(shell [ -f .env ] && sed 's/=.*//' .env)

PY := $(shell pwd)/.gym/bin/python
PIP := $(shell pwd)/.gym/bin/pip

install:
	@echo "--- Configuring environment and installing dependencies ---"
	@test -f .env || cp sample.env .env
	@if [ "$$(uname)" = "Linux" ]; then \
		sudo apt-get update && \
		sudo apt-get install -y wget zip unzip xvfb libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libatspi2.0-0 cloudflared libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2 python3.12 python3-pip python3.12-venv python3.12-dev docker-compose && \
		sudo usermod -aG docker $$USER; \
	fi
	@if [ "$$(uname)" = "Darwin" ]; then \
		command -v brew >/dev/null || \
			/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		brew install wget zip unzip python docker docker-compose; \
		echo "Install Docker Desktop manually if not installed."; \
	fi
	python3.12 -m venv .gym
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@$(PY) -m playwright install chromium
	@echo "Environment setup complete."

install-bm-webarena-image-tars:
	mkdir -p tmp_images
	@echo "--- Installing WebArena Docker images (will skip already installed) ---"
	@if ! docker image inspect shopping_final_0712:latest >/dev/null 2>&1; then \
		if [ -f tmp_images/shopping.tar ]; then \
			echo "Found partial shopping.tar file, resuming download..."; \
			wget -c -O tmp_images/shopping.tar http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar || (echo "Download failed. You can resume later by running this command again." && exit 1); \
		else \
			echo "Downloading shopping image..."; \
			wget -c -O tmp_images/shopping.tar http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar || (echo "Download failed. You can resume later by running this command again." && exit 1); \
		fi; \
		echo "Loading shopping image into Docker..."; \
		docker load < tmp_images/shopping.tar; \
		rm -f tmp_images/shopping.tar; \
	else \
		echo "Shopping image already loaded, skipping..."; \
	fi
	@if ! docker ps -a | grep -q "shopping$$"; then \
		docker run --name shopping -p 7770:80 -d shopping_final_0712 || echo "Shopping container already exists or failed to start"; \
	else \
		echo "Shopping container already exists, starting if stopped..."; \
		docker start shopping 2>/dev/null || true; \
	fi
	@if ! docker image inspect shopping_admin_final_0719:latest >/dev/null 2>&1; then \
		echo "Downloading shopping_admin image..."; \
		wget -c -O tmp_images/shopping_admin.tar http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar || (echo "Download failed. You can resume later by running this command again." && exit 1); \
		docker load < tmp_images/shopping_admin.tar; \
		rm -f tmp_images/shopping_admin.tar; \
	else \
		echo "Shopping_admin image already loaded, skipping..."; \
	fi
	@if ! docker ps -a | grep -q "shopping_admin$$"; then \
		docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719 || echo "Shopping_admin container already exists or failed to start"; \
	else \
		echo "Shopping_admin container already exists, starting if stopped..."; \
		docker start shopping_admin 2>/dev/null || true; \
	fi
	@if ! docker image inspect postmill-populated-exposed-withimg:latest >/dev/null 2>&1; then \
		echo "Downloading forum image..."; \
		wget -c -O tmp_images/forum.tar http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar || (echo "Download failed. You can resume later by running this command again." && exit 1); \
		docker load < tmp_images/forum.tar; \
		rm -f tmp_images/forum.tar; \
	else \
		echo "Forum image already loaded, skipping..."; \
	fi
	@if ! docker ps -a | grep -q "forum$$"; then \
		docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg || echo "Forum container already exists or failed to start"; \
	else \
		echo "Forum container already exists, starting if stopped..."; \
		docker start forum 2>/dev/null || true; \
	fi
	@if ! docker image inspect gitlab-populated-final-port8023:latest >/dev/null 2>&1; then \
		echo "Downloading gitlab image..."; \
		wget -c -O tmp_images/gitlab.tar http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar || (echo "Download failed. You can resume later by running this command again." && exit 1); \
		docker load < tmp_images/gitlab.tar; \
		rm -f tmp_images/gitlab.tar; \
	else \
		echo "Gitlab image already loaded, skipping..."; \
	fi
	@if ! docker ps -a | grep -q "gitlab$$"; then \
		docker run --name gitlab -p 8023:8023 -d gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start || echo "Gitlab container already exists or failed to start"; \
	else \
		echo "Gitlab container already exists, starting if stopped..."; \
		docker start gitlab 2>/dev/null || true; \
	fi
	@if ! docker ps -a | grep -q "wikipedia$$"; then \
		echo "--- Setting up Wikipedia ---"; \
		mkdir -p wikipedia_data; \
		if [ ! -f wikipedia_data/wikipedia_en_all_maxi_2022-05.zim ]; then \
			echo "Downloading Wikipedia ZIM file (95GB)..."; \
			wget -c -O wikipedia_data/wikipedia_en_all_maxi_2022-05.zim http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim; \
		fi; \
		echo "Starting wikipedia container..."; \
		docker run -d --name wikipedia -p 8888:80 -v $(shell pwd)/wikipedia_data:/data ghcr.io/kiwix/kiwix-serve:3.3.0 /data/wikipedia_en_all_maxi_2022-05.zim || echo "Wikipedia container failed to start"; \
	else \
		echo "Wikipedia container already exists, starting if stopped..."; \
		docker start wikipedia 2>/dev/null || true; \
	fi
	rm -rf tmp_images
	@echo "✅ WebArena images installation complete (skipped already installed images)"


install-bm-webarena:
	@echo "--- Starting Docker containers ---"
	@docker login --username $(DOCKER_USERNAME) --password $(DOCKER_PASSWORD)
	docker start gitlab
	docker start shopping
	docker start shopping_admin
	docker start forum
	docker start wikipedia
	@echo "--- Waiting for containers to start ---"
	@sleep 60
	@echo "--- Applying IP Table Rules if services are not accessible externally ---"
	sudo iptables -t nat -A PREROUTING -p tcp --dport 7770 -j REDIRECT --to-port 7770
	sudo iptables -t nat -A PREROUTING -p tcp --dport 7780 -j REDIRECT --to-port 7780
	sudo iptables -t nat -A PREROUTING -p tcp --dport 3000 -j REDIRECT --to-port 3000
	sudo iptables -t nat -A PREROUTING -p tcp --dport 8888 -j REDIRECT --to-port 8888
	sudo iptables -t nat -A PREROUTING -p tcp --dport 9999 -j REDIRECT --to-port 9999
	sudo iptables -t nat -A PREROUTING -p tcp --dport 8023 -j REDIRECT --to-port 8023
	@echo "--- Configuring Shopping Website ---"
	. .env && docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="$$WA_SHOPPING"
	. .env && docker exec shopping mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='$$WA_SHOPPING/' WHERE path = 'web/secure/base_url';"
	docker exec shopping /var/www/magento2/bin/magento cache:flush
	@echo "--- Configuring Shopping Admin Website ---"
	docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
	docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
	. .env && docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="$$WA_SHOPPING_ADMIN"
	. .env && docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='$$WA_SHOPPING_ADMIN/' WHERE path = 'web/secure/base_url';"
	docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
	@echo "--- Configuring GitLab Website ---"
	. .env && docker exec gitlab sed -i "s|^external_url.*|external_url $$WA_GITLAB|" /etc/gitlab/gitlab.rb
	docker exec gitlab gitlab-ctl reconfigure
	@echo "--- Testing Services ---"
	curl -s -o /dev/null -w "Shopping (7770): %{http_code}\n" http://localhost:7770
	curl -s -o /dev/null -w "Shopping Admin (7780): %{http_code}\n" http://localhost:7780
	curl -s -o /dev/null -w "GitLab (8023): %{http_code}\n" http://localhost:8023
	curl -s -o /dev/null -w "Forum (9999): %{http_code}\n" http://localhost:9999
	curl -s -o /dev/null -w "Wikipedia (8888): %{http_code}\n" http://localhost:8888
	# curl -s -o /dev/null -w "Map (3000): %{http_code}\n" http://localhost:3000
	# curl -s -o /dev/null -w "Map tile: %{http_code}\n" http://localhost:3000/tile/0/0/0.png
	@echo "WebArena setup complete."

install-bm-visualwebarena:
	@echo "--- Setting up VisualWebArena (Classifieds) ---"
	@mkdir -p classifieds_workspace
	@if [ ! -f classifieds_workspace/classifieds.zip ]; then \
		echo "Downloading Classifieds docker compose..."; \
		wget -c -O classifieds_workspace/classifieds.zip https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose.zip; \
	fi
	@if [ ! -d classifieds_workspace/classifieds_docker_compose ]; then \
		echo "Unzipping Classifieds..."; \
		unzip -o classifieds_workspace/classifieds.zip -d classifieds_workspace; \
	fi
	@echo "Configuring Classifieds..."
	# Replace default URL with localhost/hostname port 9980
	sed -i 's|CLASSIFIEDS=http://.*|CLASSIFIEDS=http://$(HOSTNAME):9980/|' classifieds_workspace/classifieds_docker_compose/docker-compose.yml
	@echo "Starting Classifieds containers..."
	cd classifieds_workspace/classifieds_docker_compose && docker compose up -d
	@echo "Waiting for DB to initialize (30s)..."
	@sleep 30
	@echo "Initializing Database Content..."
	cd classifieds_workspace/classifieds_docker_compose && docker compose exec -T db mysql -u root -ppassword osclass < mysql/osclass_craigslist.sql
	@echo "✅ VisualWebArena (Classifieds) setup complete."
	@echo "Testing Classifieds (9980)..."
	curl -s -o /dev/null -w "Classifieds (9980): %{http_code}\n" http://localhost:9980

install-bm-miniwob:
	@if [ ! -d "miniwob-plusplus" ]; then \
		git clone https://github.com/Farama-Foundation/miniwob-plusplus.git; \
	fi
	git -C miniwob-plusplus reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
	@printf "\n# Benchmark 4: MiniWoB++\nMINIWOB_URL=\"file://\$${PWD}/miniwob-plusplus/miniwob/html/miniwob/\"" >> .env
	@echo "MiniWob++ setup complete."

demo:
	@echo "--- Running demo agent with tasks ---"
	@if [ ! -f .env ]; then echo "Error: .env file not found. Please create .env file."; exit 1; fi
	cd agents && \
		. ../.env && \
		if [ -z "$(TASKS)" ]; then \
			echo "No TASKS specified — running open-ended demo agent"; \
			xvfb-run -a $(PY) green_agent.py; \
		else \
			for TASK in $(TASKS); do \
				echo "\n=== Working on task: $$TASK ==="; \
				BENCHMARK=$$(echo $$TASK | cut -d. -f1); \
				[ "$$BENCHMARK" = "webarena" ] && export WEB_ARENA_DATA_DIR="$$WEB_ARENA_DATA_DIR"; \
				[ "$$BENCHMARK" = "visualwebarena" ] && export DATASET="$$DATASET"; \
				xvfb-run -a $(PY) green_agent.py --task $$TASK; \
				echo "\n✅ Finished task: $$TASK"; \
			done; \
		fi

demo-bm-webarena:
	make demo TASKS="webarena.4"

demo-bm-visualwebarena:
	make demo TASKS="visualwebarena.398"

demo-bm-workarena:
	make demo TASKS="workarena.servicenow.order-standard-laptop"

demo-bm-miniwob:
	make demo TASKS="miniwob.click-test"

demo-bm-assistantbench:
	make demo TASKS="assistantbench.validation.3"

demo-bm-weblinx:
	make demo TASKS="weblinx.klatidn.1"

GREEN_AGENT_PATH ?= demo_agent/agent.py
GREEN_MAX_STEPS ?= 50

green-workarena:
	@if [ -z "$(TASK)" ]; then \
		echo "Please provide TASK=workarena.servicenow.<task-name>"; \
		exit 1; \
	fi
	@if [ ! -f .env ]; then echo "Error: .env file not found. Please create .env file."; exit 1; fi
	@echo "--- Running Green Evaluator on WorkArena task: $(TASK) ---"
	. .env && \
	$(PY) green_evaluator.py \
		--agent_path $(GREEN_AGENT_PATH) \
		--task $(TASK) \
		--max_steps $(GREEN_MAX_STEPS)

green-webarena:
	@if [ -z "$(TASK)" ]; then \
		echo "Please provide TASK=webarena.<task-id> (e.g., TASK=webarena.4)"; \
		exit 1; \
	fi
	@if [ ! -f .env ]; then echo "Error: .env file not found. Please create .env file."; exit 1; fi
	@echo "--- Running Green Evaluator on WebArena task: $(TASK) ---"
	@echo "Note: WebArena requires Docker containers running on your GCP VM"
	@echo "Make sure WA_* environment variables are set in .env pointing to your VM"
	. .env && \
	$(PY) green_evaluator.py \
		--agent_path $(GREEN_AGENT_PATH) \
		--task $(TASK) \
		--max_steps $(GREEN_MAX_STEPS)

test-core:
	@echo "--- Running core tests ---"
	@if [ ! -f .env ]; then echo "Error: .env file not found. Please create .env file."; exit 1; fi
	. .env && $(PY) -m pytest -n auto ./tests/core

test-green:
	@echo "--- Running Green Evaluator tests ---"
	@if [ ! -f .env ]; then echo "Error: .env file not found. Please create .env file."; exit 1; fi
	. .env && $(PY) -m pytest -v ./tests/green_evaluator

test-benchmarks:
	@echo "--- Running tests for installed benchmarks only ---"
	@if [ ! -f .env ]; then echo "Error: .env file not found. Please create .env file."; exit 1; fi
	. .env && $(PY) -m pytest -n auto ./tests/core ./tests/green_evaluator ./tests/miniwob ./tests/assistantbench ./tests/experiments -v

test-all:
	@echo "--- Running all tests ---"
	@if [ ! -f .env ]; then echo "Error: .env file not found. Please create .env file."; exit 1; fi
	. .env && $(PY) -m pytest -n auto ./tests

clean-bm-miniwob:
	@echo "--- Cleaning MiniWoB++ installation ---"
	rm -rf miniwob-plusplus
	@echo "MiniWoB++ installation cleaned!"

clean-bm-webarena:
	@echo "--- Cleaning WebArena installation ---"
	@docker rm -f shopping shopping_admin forum gitlab wikipedia 2>/dev/null || true
	@docker image rm \
		shopping_final_0712 \
		shopping_admin_final_0719 \
		postmill-populated-exposed-withimg \
		gitlab-populated-final-port8023 \
		ghcr.io/kiwix/kiwix-serve:3.3.0 \
		2>/dev/null || true
	rm -rf wikipedia_data
	@echo "WebArena cleaned"

clean-bm-visualwebarena:
	@echo "--- Cleaning VisualWebArena ---"
	@if [ -d classifieds_workspace/classifieds_docker_compose ]; then \
		cd classifieds_workspace/classifieds_docker_compose && docker compose down -v; \
	fi
	rm -rf classifieds_workspace
	@echo "VisualWebArena cleaned"

help:
	@echo "Available targets:"
	@echo "  install                          - Install project dependencies"
	@echo "  install-bm-miniwob               - Install MiniWoB++"
	@echo "  install-bm-webarena-image-tars   - Install WebArena Docker image tars"
	@echo "  install-bm-webarena              - Install and configure WebArena"
	@echo "  install-bm-visualwebarena        - Install VisualWebArena (Classifieds)"
	@echo "  demo                             - Run demo agent"
	@echo "  demo-bm-miniwob                  - Run MiniWoB++ demo"
	@echo "  demo-bm-webarena                 - Run WebArena demo"
	@echo "  demo-bm-visualwebarena           - Run VisualWebArena demo"
	@echo "  demo-bm-workarena                - Run WorkArena demo"
	@echo "  demo-bm-assistantbench           - Run AssistantBench demo"
	@echo "  demo-bm-weblinx                  - Run WebLinx demo"
	@echo "  green-workarena                  - Run Green Evaluator on a WorkArena task (TASK=...)"
	@echo "  green-webarena                   - Run Green Evaluator on a WebArena task (TASK=...)"
	@echo "  test-core                        - Run core tests"
	@echo "  test-green                       - Run Green Evaluator tests"
	@echo "  test-benchmarks                  - Run tests for installed benchmarks only"
	@echo "  test-all                         - Run all tests (includes WebArena/VisualWebArena)"
	@echo "  clean-bm-miniwob                 - Remove MiniWoB++ installation"
	@echo "  clean-bm-webarena                - Remove WebArena containers and data"
	@echo "  clean-bm-visualwebarena          - Remove VisualWebArena (Classifieds)"
	@echo "  clean-bm-weblinx                 - Remove WebLinx caches and artifacts"
	@echo "  help                             - Show this help message"

.PHONY: install install-bm-miniwob install-bm-webarena-image-tars install-bm-webarena install-bm-visualwebarena demo demo-bm-miniwob demo-bm-webarena demo-bm-visualwebarena
.PHONY: demo-bm-workarena demo-bm-assistantbench demo-bm-weblinx green-workarena green-webarena test-core test-green test-installed test-all clean-bm-miniwob clean-bm-webarena clean-bm-visualwebarena help
