import jsonlines
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default='tmp')
parser.add_argument("--lang-id", type=str)
parser.add_argument("--model-name", type=str)
args = parser.parse_args()

## make dir
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

device="cuda"
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map=device)

## zero-shot
def zero_shot_prompt(obj):    
    instruction = "Given the following passage, query, and answer choices, output the letter corresponding to the correct answer."
    passage = obj['flores_passage']
    query = obj['question']
    A = obj['mc_answer1']
    B = obj['mc_answer2']
    C = obj['mc_answer3']
    D = obj['mc_answer4']
    prompt = f"{instruction}\n###\nPassage:\n{passage}\n###\nQuery:\n{query}\n###\nChoices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n###\nAnswer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    outputs_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs_decoded

with jsonlines.open(output_dir / "answers.jsonl", "w", flush=True) as writer:
    with jsonlines.open(f"data/Belebele/{args.lang_id}.jsonl") as reader:
        for obj in reader:
            zero_shot_answer = zero_shot_prompt(obj)
            writer.write({"link": obj["link"],
                          "question_number": obj["question_number"],
                          "correct_answer": obj["correct_answer_num"], 
                          "answer":zero_shot_answer})


