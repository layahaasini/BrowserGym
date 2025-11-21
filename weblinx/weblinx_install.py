from datasets import load_dataset
from huggingface_hub import snapshot_download

# Load the validation split
valid = load_dataset("McGill-NLP/weblinx", split="validation")

# Download the input templates and use the LLaMA one
snapshot_download(
    "McGill-NLP/WebLINX", repo_type="dataset", allow_patterns="templates/*", local_dir="."
)
with open('templates/llama.txt') as f:
    template = f.read()

# To get the input text, simply pass a turn from the valid split to the template
turn = valid[0]
turn_text = template.format(**turn)