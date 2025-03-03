from datasets import load_dataset
import jsonlines
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import json
from pathlib import Path
from utils.dataset_locations import mc2_location
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default='tmp')
parser.add_argument("--lang", type=str, help="one of tibetan, kazakh, uyghur, mongolian")
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
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)

def prompt(text):    
    instruction = """
    Identify the language of the given text. Output the English name of the language. Do not add explanation.
    Example:

    Text: 地元メディアの報道によると、空港の消防車が対応中に横転したということです。
    Language: Japanese

    Text: 그 조종사는 비행 중대장 딜로크리트 패타비로 확인되었다.
    Language: Korean

    """

    prompt = f"{instruction}\n\nText: {text}\nLanguage: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    outputs_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs_decoded

datafiles_location = {
    "tibetan": "bo-crawl-only-release-20231112.jsonl", 
    "kazakh": "kk-crawl-only-release-20231112.jsonl",
    "mongolian": "mn-crawl-only-release-20231127.jsonl",
    "uyghur": "ug-crawl-only-release-20231112.jsonl" 
                              }
mc2_dataset = load_dataset(
    mc2_location, 
    data_files=datafiles_location[args.lang],
    streaming=True
)

with jsonlines.open(output_dir / "answers.jsonl", "w", flush=True) as writer:
    for i, obj in enumerate(mc2_dataset['train']):
        answer = prompt(obj['text'])
        writer.write({"url": obj["url"],
                        "title": obj["title"],
                        "answer":answer})

        if i == 100: break
        
