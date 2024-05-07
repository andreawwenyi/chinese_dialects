import sys
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
device = 'auto'
loss_function = CrossEntropyLoss()

## make dir
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

## load model
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map=device)
model.eval()

def eval_ppl(encodings):
    n_toks = encodings.input_ids.size(1)
    past_key_values = None
    nlls = []
    for loc in range(0, n_toks-1):
        input_ids = encodings.input_ids[:, loc:loc+1].to(device)
        labels = encodings.input_ids[:, loc+1:loc+2]
        
        with torch.no_grad():
            # https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llama#transformers.LlamaModel
            outputs = model(input_ids,
                            past_key_values = past_key_values,
                            use_cache = True, # return past_key_values
                            ) # type(outputs): 'transformers.modeling_outputs.CausalLMOutputWithPast'
            
            logits = outputs.logits.reshape(-1, model.config.vocab_size)
            nll = loss_function(logits.detach().cpu(), labels.view(-1))
            # past_seq_len = outputs.past_key_values[0][0].shape[2]
            # if past_seq_len > window-1:
            #     past_key_values = list()
            #     # slice past_key_values
            #     for k, v in outputs.past_key_values: # each tuple has 2 tensors
            #         k_slice = k[:, :, -window+1:, :]
            #         v_slice = v[:, :, -window+1:, :]
            #         past_key_values.append((k_slice, v_slice))
            # else:
            #     past_key_values = outputs.past_key_values
        nlls.append(nll)
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


with open(args.input_file, "r") as f: 
    texts = f.readlines()

f = open(output_dir / "logs.txt", "w")
f.write("n_toks\tppl\n")
for t in texts:
    encodings = tokenizer(t, return_tensors="pt")
    # ppl = eval_ppl(encodings)
    ppl = 0
    n_toks = encodings.input_ids.size(1)
    print(f"{n_toks}\t{ppl}", file=f, flush=True)
f.close()