import os
from huggingface_hub import snapshot_download

project_dir = os.environ.get("WEBLINX_PROJECT_DIR")
data_dir = os.environ.get("WEBLINX_DATA_DIR")

if project_dir is None or data_dir is None:
    raise RuntimeError("WEBLINX_PROJECT_DIR and WEBLINX_DATA_DIR must be set in .env")

os.makedirs(project_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

snapshot_download(
    repo_id="McGill-NLP/WebLINX-full",
    repo_type="dataset",
    local_dir=data_dir,
)
