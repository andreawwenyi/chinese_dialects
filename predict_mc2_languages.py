import pandas as pd
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
from prompt_chat_models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default='tmp')
parser.add_argument("--lang", type=str, help="one of tibetan, kazakh, uyghur, mongolian")
parser.add_argument("--model-path", type=str)
args = parser.parse_args()

## make dir
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

## read models.csv
models = pd.read_csv("models.csv")
models['system_prompt'] = models['system_prompt'].fillna("")

is_instruct_model = models[models['model_path_hf'] == args.model_path]["instruction_tuned"].values[0]
print("is_instruct_model: ", is_instruct_model)
if is_instruct_model:
    instruct_model = load_model(args.model_path)
if not is_instruct_model:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()

def make_prompt(text):
    instruction = """
    Identify the language of the given text. Output the English name of the language. Be concise.
    Example:

    Text: 地元メディアの報道によると、空港の消防車が対応中に横転したということです。
    Language: Japanese

    Text: 그 조종사는 비행 중대장 딜로크리트 패타비로 확인되었다.
    Language: Korean

    """

    prompt = f"{instruction}Text: {text}\nLanguage: "
    return prompt

def run_base_model(text):    
    prompt = make_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    # Get just the new tokens by finding where the input ends
    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_length:]
    # Decode only the new tokens
    outputs_decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return outputs_decoded

def prompt_model(text):
    if is_instruct_model:
        system_prompt = models[models['model_path_hf'] == args.model_path]['system_prompt'].values[0]
        user_prompt = make_prompt(text)
        response = instruct_model.prompt_chat_model(system_prompt, user_prompt)
        return response
    else:
        response = run_base_model(text)
        return response

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
        answer = prompt_model(obj['text'])
        writer.write({
            "url": obj["url"],
            "title": obj["title"],
            "answer":answer
        })

        if i == 100: break
        
