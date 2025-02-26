from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from pathlib import Path
from utils.ppl import eval_ppl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default='tmp')
parser.add_argument("--input-file", type=str)
parser.add_argument("--model-name", type=str)
args = parser.parse_args()

## make dir
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

## device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

## load model
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map=device)
model.eval()

with open(args.input_file, "r") as f: 
    texts = f.readlines()

f = open(output_dir / "logs_2.txt", "w")
f.write("n_toks\tnll_sum\tppl\n")

for i, t in enumerate(texts):
    encodings = tokenizer(t, return_tensors="pt")
    n_toks = encodings.input_ids.size(1)
    nll_sum, ppl = eval_ppl(model, encodings)
    print(f"{n_toks}\t{nll_sum}\t{ppl}", file=f, flush=True)
f.close()