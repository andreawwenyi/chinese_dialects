import jsonlines
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import json
from pathlib import Path
import argparse
from transformers.generation.utils import GenerationConfig

## device
class InstructModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    def prompt_chat_model(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = ""

        return response

class QwenInstruct(InstructModel):
    def prompt_chat_model(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # tokenize using apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class YiInstruct(InstructModel):
    def prompt_chat_model(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to(self.model.device), eos_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

class InternLMInstruct(InstructModel):
    def prompt_chat_model(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        tokenized_messages = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(tokenized_messages, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_messages, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class BaichuanInstruct(InstructModel):
    def prompt_chat_model(self, system_prompt, user_prompt):
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.model.chat(self.tokenizer, messages)
        return response

class LlamaInstruct(InstructModel):
    def prompt_chat_model(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        output_ids = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        print(response)
        return response

class MistralInstruct(InstructModel):
    def prompt_chat_model(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=256)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

class GemmaInstruct(InstructModel):
    def prompt_chat_model(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(self.model.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=256)
        response = self.tokenizer.decode(outputs[0])

        return response

class OlmoInstruct(InstructModel):
    def prompt_chat_model(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        raise

class DeepSeekInstruct(InstructModel):
    def prompt_chat_model(self, system_prompt, user_prompt):
        system_prompt = ""
        """
        We recommend adhering to the following configurations when utilizing the DeepSeek-R1 series models, including benchmarking, to achieve the expected performance:

        Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent endless repetitions or incoherent outputs.
        Avoid adding a system prompt; all instructions should be contained within the user prompt.
        For mathematical problems, it is advisable to include a directive in your prompt such as: "Please reason step by step, and put your final answer within \boxed{}."
        When evaluating model performance, it is recommended to conduct multiple tests and average the results.
        Additionally, we have observed that the DeepSeek-R1 series models tend to bypass thinking pattern (i.e., outputting "<think>\n\n</think>") when responding to certain queries, which can adversely affect the model's performance. To ensure that the model engages in thorough reasoning, we recommend enforcing the model to initiate its response with "<think>\n" at the beginning of every output.

        """

def load_model(model_path):
    model_name = model_path.split("/")[-1]
    if model_name.lower().startswith("qwen"):
        instruct_model = QwenInstruct(model_path)
    elif model_name.lower().startswith("yi"):
        instruct_model = YiInstruct(model_path)
    elif model_name.lower().startswith("olmo"):
        instruct_model = OlmoInstruct(model_path)
    elif model_name.lower().startswith("internlm"):
        instruct_model = InternLMInstruct(model_path)
    elif model_name.lower().startswith("baichuan"):
        instruct_model = BaichuanInstruct(model_path)
    elif model_name.lower().startswith("meta-llama"):
        instruct_model = LlamaInstruct(model_path)
    elif model_name.lower().startswith("mistral"):
        instruct_model = MistralInstruct(model_path)
    elif model_name.lower().startswith("gemma"):
        instruct_model = GemmaInstruct(model_path)
    elif model_name.lower().startswith("olmo"):
        instruct_model = OlmoInstruct(model_path)
    elif model_name.lower().startswith("deepseek-r1"):
        instruct_model = DeepSeekInstruct(model_path)
    else:
        raise Exception(f"Unknown model {model_path}")
    return instruct_model


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    instruct_model = load_model(model_path)
    system_prompt = "You are a helpful assistant."
    user_prompt = "Who is Toni Morrison?"
    response = instruct_model.prompt_chat_model(system_prompt, user_prompt)
    print(response)
    

