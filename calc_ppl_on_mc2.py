from datasets import load_dataset
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
parser.add_argument("--lang", type=str, help="one of bo, kk, mn, ug")
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

datafiles_location = {"bo": "bo-crawl-only-release-20231112.jsonl", # Tibetan
                      "kk": "kk-crawl-only-release-20231112.jsonl", # Kazakh
                      "mn": "mn-crawl-only-release-20231127.jsonl", # Mongolian
                      "ug": "ug-crawl-only-release-20231112.jsonl" # Uyghur
                              }
mc2_dataset = load_dataset("pkupie/mc2_corpus", 
                          data_files=datafiles_location[args.lang]
                              )


f = open(output_dir / "logs_2.txt", "w")
f.write("n_toks\tppl\n")

for t in mc2_dataset['train']['text']:
    encodings = tokenizer(t, return_tensors="pt")
    n_toks = encodings.input_ids.size(1)
    ppl = eval_ppl(model, encodings)
    print(f"{n_toks}\t{ppl}", file=f, flush=True)
f.close()
