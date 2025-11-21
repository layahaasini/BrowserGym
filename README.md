# BrowserGym: Green Agent

BrowserGym provides a unified ecosystem for training and evaluating web agents across realistic browser environments and benchmarks.

---

## Benchmarks Used

BrowserGym integrates multiple web automation and reasoning benchmarks:

- [**MiniWoB**](https://miniwob.farama.org/) — [browsergym/miniwob/README.md](browsergym/miniwob/README.md)  
- [**WebArena**](https://webarena.dev/) — [browsergym/webarena/README.md](browsergym/webarena/README.md)  
- [**VisualWebArena**](https://jykoh.com/vwa) — [browsergym/visualwebarena/README.md](browsergym/visualwebarena/README.md)  
- [**WorkArena**](https://github.com/ServiceNow/WorkArena) — enterprise-grade knowledge-work tasks  
- [**AssistantBench**](https://github.com/oriyor/assistantbench) — [browsergym/assistantbench/README.md](browsergym/assistantbench/README.md)  
- [**WebLINX**](https://github.com/McGill-NLP/weblinx) — static web interaction dataset  

---

## Setup

### 1. Create a compute instance
- gcp, ec2, etc (AWS is recommended if you want to have an easier setup for benchmark 2, but any work fine)
- recommended system: ubuntu
- ssh into the instance with ssh in browser or ssh -i <private ssh key file> <user>@<external_ip_address>
--> can run whoami to find out user id

### 2. Configure OpenAI API Key
- get openaikey 
- in .env, export it
```
OPENAI_API_KEY="<key beginning with sk-proj- ...>"
```

### 3. Clone the Repository
~~~bash
git clone <repository-url>
cd BrowserGym
~~~

### 4. Set Up The Environment
- make sure to change lines 6-7 depending on what version you have

- install make if you have never used it
```
# Ubuntu/Debian
sudo apt update && sudo apt install make -y

# Fedora/RHEL
sudo dnf install make -y

# Arch Linux
sudo pacman -S make

# macOS (via Homebrew)
brew install make
```
~~~bash
make install
~~~

### 5. Set up the benchmarks
Benchmark 1: MiniWob++
```make benchmark1```
- if done correctly, you should see this in ur .env
```
MINIWOB_URL="file:///<PWD>/BrowserGym/miniwob-plusplus/miniwob/html/miniwob/"
```

Benchmark 2: WebArena
- make baseurl = port or localhost in .env
```
BASE_URL="http://<FILL IN EXTERNAL IP ADDRESS>"
BASE_URL="http://localhost"
```
- configure docker username/pass in .env
```
DOCKER_USERNAME=<FILL IN USERNAME>
DOCKER_PASSWORD=<FILL IN PASSWORD>
```
- configure instance as needed
if using aws, 
--> use ami to start new ec2 instance (EC2 console >> AMIs) - further instructions found [in recommended path](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#individual-website) and ```make install-benchmark2```
--> troubleshooting? make sure region set to ohio and if image names are not configured, mkae sure they appear in docker images
if using another instance,
```
make install-benchmark2-image-tars
```
- install the benchmark
```
make install-benchmark2
```

Benchmark 3: VisualWebArena
- TBD

Benchmark 4: WorkArena
- TBD

Benchmark 5: AssistantBench
- no set up required

Benchmark 6: Weblinx
- add the following into your .env
```
WEBLINX_ROOT=${PWD}/weblinx
WEBLINX_PROJECT_DIR=${PWD}/weblinx/modeling
WEBLINX_DATA_DIR=${PWD}/weblinx/modeling/wl_data
```
- run make install-benchmark5

### Sanity Check
if all works as expected, ur .env should look something like this
```
# OpenAI
OPENAI_API_KEY="sk-proj-<FILL IN KEY>"

# Benchmark 1: MiniWoB++
MINIWOB_URL="file:///<FILL IN PWD>/BrowserGym/miniwob-plusplus/miniwob/html/miniwob/"

# Benchmark 2: WebArena (AWS or other)
BASE_URL="http://<FILL IN EXTERNAL IP ADDRESS>" or "http://localhost"
DOCKER_USERNAME=<FILL IN USERNAME>
DOCKER_PASSWORD=<FILL IN PASSWORD>
SHOPPING_IMAGE_NAME="shopping_final_0712"
SHOPPING_ADMIN_IMAGE_NAME="shopping_admin_final_0719"
GITLAB_IMAGE_NAME="gitlab-populated-final-port8023" 
FORUM_IMAGE_NAME="postmill-populated-exposed-withimg" 
WA_SHOPPING="${BASE_URL}:7770"
WA_SHOPPING_ADMIN="${BASE_URL}:7771"
WA_REDDIT="${BASE_URL}:9999"
WA_GITLAB="${BASE_URL}:9980"
WA_WIKIPEDIA="${BASE_URL}:8888"
WA_MAP="${BASE_URL}:4444"
WA_HOMEPAGE="${BASE_URL}:4399"
VWA_CLASSIFIEDS="${BASE_URL}:9980"
VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
# Optional WebArena Config
MAP_BACKEND_IP="${BASE_URL}:3000"
# Optional (if using WebArena's Full Reset feature)
# WA_FULL_RESET="${BASE_URL}:7565"

# Benchmark 3: VisualWebArena
DATASET=visualwebarena
VWA_CLASSIFIEDS=http://localhost:9980
VWA_CLASSIFIEDS_RESET_TOKEN=4b61655535e7ed388f0d40a93600254c
VWA_SHOPPING=http://localhost:7770
VWA_REDDIT=http://localhost:9999
VWA_WIKIPEDIA=http://localhost:8888
VWA_HOMEPAGE=http://localhost:4399

# Benchmark 4: WorkArena
SNOW_INSTANCE_URL=<<FILL IN INSTANCE URL>>
SNOW_INSTANCE_UNAME=<FILL IN INSTANCE USERNAME>>
SNOW_INSTANCE_PWD=<FILL IN INSTANCE PASSWORD>>

# Benchmark 5: Weblinx
WEBLINX_ROOT=${PWD}/weblinx
WEBLINX_PROJECT_DIR=${PWD}/weblinx/modeling
WEBLINX_DATA_DIR=${PWD}/weblinx/modeling/wl_data

# AgentBeats
AGENT_HOST=<FILL IN EXTERNAL IP ADDRESS> or localhost if not running on SSH on VM
AGENT_PORT=8000
LAUNCHER_HOST=<FILL IN EXTERNAL IP ADDRESS> or localhost
LAUNCHER_PORT=8080
```

---

## Usage

### 1. Activate Environment
~~~bash
# Linux/macOS
source .gym/bin/activate

# Windows (PowerShell)
.gym\Scripts\activate
~~~

### 2. Run Demo Tasks

Use `make demo` to run one or more tasks across benchmarks.

#### MiniWoB
~~~bash
make demo TASKS="miniwob.click-test"
make demo TASKS="miniwob.choose-date miniwob.click-dialog"
~~~

#### WorkArena
~~~bash
make demo TASKS="workarena.servicenow.order-standard-laptop"
make demo TASKS="workarena.servicenow.order-ipad-pro"
~~~

#### WebArena
~~~bash
make demo TASKS="webarena.4"
make demo TASKS="webarena.310"
~~~

#### VisualWebArena
~~~bash
make demo TASKS="visualwebarena.398"
make demo TASKS="visualwebarena.721"
~~~

#### AssistantBench
~~~bash
make demo TASKS="assistantbench.validation.3"
~~~

#### WebLinx
~~~bash
make demo TASKS="assistantbench.validation.3"
~~~

#### Run Multiple Tasks at Once
~~~bash
make demo TASKS="miniwob.click-test webarena.4 visualwebarena.398 workarena.servicenow.order-standard-laptop assistantbench.validation.3"
~~~

#### Show all available options
~~~bash
python demo_agent/run_demo.py --help
~~~

### 3. Run Tests
~~~bash
make test-core
~~~

---

## AgentBeats Integration

### 1. Install `wget` (required for AgentBeats)

Install `wget` depending on your OS:

**macOS**
~~~bash
brew install wget
~~~

**Ubuntu / Debian**
~~~bash
sudo apt update && sudo apt install wget -y
~~~

**Fedora / RHEL**
~~~bash
sudo dnf install wget -y
~~~

**Arch Linux**
~~~bash
sudo pacman -S wget
~~~

**Windows (with Chocolatey)**
~~~bash
choco install wget
~~~

Then install AgentBeats:
~~~bash
make install-agentbeats
~~~

### 2. Register agent and battle
~~~bash
make register-agent
make register-battle
~~~

> Edit metadata in your agent/battle configuration before registering if needed.

---

## Ecosystem

- [AgentLab](https://github.com/ServiceNow/AgentLab): Run agents on benchmarks and collect traces  
- [WorkArena(++)](https://github.com/ServiceNow/WorkArena): Web agents for enterprise workflows  
- [WebArena](https://github.com/web-arena-x/webarena): Realistic web task benchmark  
- [VisualWebArena](https://github.com/web-arena-x/visualwebarena): Visual reasoning web tasks  
- [MiniWoB(++)](https://miniwob.farama.org/): Synthetic web microtasks  
- [WebLINX](https://github.com/McGill-NLP/weblinx): Real-world web interaction dataset  
- [AssistantBench](https://github.com/oriyor/assistantbench): Open web task benchmark  
- [DoomArena](https://github.com/ServiceNow/DoomArena): Web agent security testing framework  

---

## Contributors

[![BrowserGym contributors](https://contrib.rocks/image?repo=ServiceNow/BrowserGym&max=2000)](https://github.com/ServiceNow/BrowserGym/graphs/contributors)

---

## Citation

If you use BrowserGym, please cite:

```bibtex
@article{
    chezelles2025browsergym,
    title={The BrowserGym Ecosystem for Web Agent Research},
    author={Thibault Le Sellier de Chezelles and Maxime Gasse and Alexandre Lacoste and Massimo Caccia and Alexandre Drouin and L{\'e}o Boisvert and Megh Thakkar and Tom Marty and Rim Assouel and Sahar Omidi Shayegan and Lawrence Keunho Jang and Xing Han L{\`u} and Ori Yoran and Dehan Kong and Frank F. Xu and Siva Reddy and Graham Neubig and Quentin Cappart and Russ Salakhutdinov and Nicolas Chapados},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=5298fKGmv3},
    note={Expert Certification}
}

@inproceedings{workarena2024,
    title = {{W}ork{A}rena: How Capable are Web Agents at Solving Common Knowledge Work Tasks?},
    author = {Drouin, Alexandre and Gasse, Maxime and Caccia, Massimo and Laradji, Issam H. and Del Verme, Manuel and Marty, Tom and Vazquez, David and Chapados, Nicolas and Lacoste, Alexandre},
    booktitle = {Proceedings of the 41st International Conference on Machine Learning},
    pages = {11642--11662},
    year = {2024},
    publisher = {PMLR},
    url = {https://proceedings.mlr.press/v235/drouin24a.html},
}
