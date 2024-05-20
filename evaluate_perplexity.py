from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from torch.nn import CrossEntropyLoss
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default='tmp')
parser.add_argument("--input-file", type=str)
parser.add_argument("--model-name", type=str)
args = parser.parse_args()
device = 'cuda'
loss_function = CrossEntropyLoss()

## make dir
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

## load model
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map=device)
model.eval()

def eval_ppl(encodings):  
    """
    source: https://huggingface.co/docs/transformers/perplexity
    """
    max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        print(trg_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    nll_sum = torch.stack(nlls).sum()
    print(torch.stack(nlls).sum())
    print(torch.stack(nlls).mean())
    ppl = torch.exp(torch.stack(nlls).mean())
    return nll_sum, ppl
    

with open(args.input_file, "r") as f: 
    texts = f.readlines()

f = open(output_dir / "logs_2.txt", "w")
f.write("n_toks\tnll_sum\tppl\n")

for i, t in enumerate(texts):
    encodings = tokenizer(t, return_tensors="pt")
    n_toks = encodings.input_ids.size(1)
    nll_sum, ppl = eval_ppl(encodings)
    print(f"{n_toks}\t{nll_sum}\t{ppl}", file=f, flush=True)
f.close()