.SILENT:

include .env
export $(shell sed 's/=.*//' .env)

PY := $(shell pwd)/.gym/bin/python
PIP := $(shell pwd)/.gym/bin/pip

install:
	@echo "--- Configuring environment and installing dependencies ---"
	@if [ "$$(uname)" = "Linux" ]; then \
		sudo apt-get update && \
		sudo apt-get install -y wget xvfb libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libatspi2.0-0 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2 docker.io python3-pip python3-venv && \
		sudo apt install -y python3-pip python3-venv docker-compose && \
		sudo usermod -aG docker $$USER; \
	else \
		echo "Skipping apt-get (not Linux)."; \
	fi
	python3 -m venv .gym
	@$(PIP) install --upgrade pip
	@$(PIP) install --break-system-packages -r requirements.txt || true  # Run this to handle errors if they occur
	@$(PIP) install -r requirements.txt
	@$(PY) -m playwright install chromium
	@echo "Environment setup complete."

install-benchmark1:
	@if [ ! -d "miniwob-plusplus" ]; then \
		git clone https://github.com/Farama-Foundation/miniwob-plusplus.git; \
	fi
	git -C miniwob-plusplus reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
	echo "MINIWOB_URL=\"file://$(shell pwd)/miniwob-plusplus/miniwob/html/miniwob/\"" >> .env
	@echo "MiniWob++ setup complete."

install-benchmark2-image-tars:
	set -e
	echo "== Loading Shopping Website =="
	wget -O - http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar | docker load
	docker run --name shopping -p 7770:80 -d shopping_final_0712
	echo "== Loading Shopping Admin Website =="
	wget -O - http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar | docker load
	docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
	echo "== Loading Reddit / Forum Website =="
	wget -O - http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar | docker load
	docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg
	echo "== Loading GitLab Website =="
	wget -O - http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar | docker load
	docker run --name gitlab -p 8023:8023 -d gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
	echo "== Starting Wikipedia WITHOUT downloading .zim locally =="
	docker run -d \
	--name wikipedia \
	-p 8888:80 \
	--mount type=tmpfs,destination=/data \
	ghcr.io/kiwix/kiwix-serve:3.3.0 \
	https://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim
	@echo "WebArena image tars uploaded to Docker."

install-benchmark2:
	@echo "--- Starting Docker containers ---"
	@docker login --username $(DOCKER_USERNAME) --password $(DOCKER_PASSWORD)
	docker start gitlab
	docker start shopping
	docker start shopping_admin
	docker start forum
	docker start kiwix33
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
	docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$(HOSTNAME):7770"
	docker exec shopping mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value="http://$(HOSTNAME):7770/" WHERE path = "web/secure/base_url";'
	docker exec shopping /var/www/magento2/bin/magento cache:flush
	@echo "--- Configuring Shopping Admin Website ---"
	docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
	docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
	docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$(HOSTNAME):7780"
	docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value="http://$(HOSTNAME):7780/" WHERE path = "web/secure/base_url";'
	docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
	@echo "--- Configuring GitLab Website ---"
	docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$(HOSTNAME):8023'|" /etc/gitlab/gitlab.rb
	docker exec gitlab gitlab-ctl reconfigure
	@echo "--- Testing Services ---"
	curl -s -o /dev/null -w "Shopping (7770): %{http_code}\n" http://$(HOSTNAME):7770
	curl -s -o /dev/null -w "Shopping Admin (7780): %{http_code}\n" http://$(HOSTNAME):7780
	curl -s -o /dev/null -w "Forum (9999): %{http_code}\n" http://$(HOSTNAME):9999
	curl -s -o /dev/null -w "Wikipedia (8888): %{http_code}\n" http://$(HOSTNAME):8888
	curl -s -o /dev/null -w "Map (3000): %{http_code}\n" http://$(HOSTNAME):3000
	curl -s -o /dev/null -w "GitLab (8023): %{http_code}\n" http://$(HOSTNAME):8023
	curl -s -o /dev/null -w "Map tile: %{http_code}\n" http://$(HOSTNAME):3000/tile/0/0/0.png
	@echo "WebArena setup complete."

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
	@echo "  install                          - Install project dependencies"
	@echo "  install-benchmark1               - Install MiniWob++"
	@echo "  install-benchmark2-image-tars    - Install WebArena image tars if not configuring with AWS"
	@echo "  install-benchmark2               - Install WebArena"
	@echo "  install-agentbeats               - Install agentbeats"
	@echo "  register-agent                   - Register your agent onto the Agentbeats platform"
	@echo "  register-battle                  - Register your agent for an Agentbeats battle"
	@echo "  demo                             - Run demo agent"
	@echo "  test-core                        - Run core tests"
	@echo "  clean-miniwob                    - Remove MiniWoB++ directory"
	@echo "  help                             - Show this help message"

.PHONY: install install-benchmark1 install-benchmark2-image-tars install-benchmark2 install-agentbeats register-agent register-battle demo test-core clean-miniwob help
