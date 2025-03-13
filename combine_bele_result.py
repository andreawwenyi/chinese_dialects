import jsonlines
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
from jsonlines.jsonlines import InvalidLineError

langs = pd.read_csv("langs.csv")
langs = langs[langs["lang_code_bele"].notnull()]
langs["lang_code"] = langs["lang_code_bele"]
models = pd.read_csv("models.csv")
models["model_name"] = models["model_path_hf"].apply(lambda t: t.split("/")[-1])
model_order = models["model_name"]

correct_answers = dict()    
for lang_code in langs['lang_code'].values:
    correct_answers[lang_code] = list()
    data_file = f"./data/Belebele/{lang_code}.jsonl"
    answer_map = {"1": "A", "2": "B", "3": "C", "4":"D"}
    with jsonlines.open(data_file) as reader:
        for obj in reader:
            correct_answers[lang_code].append(answer_map[obj["correct_answer_num"]])

def extract_answer_choice(model_name, answer_output):
    answer_output = answer_output.replace("\n", "")
    pattern=r"###Answer:.*?([ABCD])"
    match = re.search(pattern, answer_output)
    answer_choice = match.group(1) if match else "X"
    return answer_choice

def extract_answer_from_chat_templates(model_name, answer_output):
    if model_name.lower().startswith("mistral"):
        pattern = r"\[/INST\]\s*\(([ABCD])\)"
        match = re.search(pattern, answer_output)
        answer_choice = match.group(1).upper() if match else "X"
        return answer_choice
    elif (model_name.lower().startswith("yi")) or (model_name.lower().startswith("qwen")):
        if (len(answer_output) == 1):
            answer_choice = answer_output
        else:
            pattern = r"\(([ABCD])\)"
            match = re.search(pattern, answer_output)
            answer_choice = match.group(1).upper() if match else "X"
        return answer_choice
    elif model_name.lower().startswith("internlm"):
        answer_choice = "X"
        
        if (len(answer_output) == 1):
            answer_choice = answer_output
        else:
            patterns = [r"答案是.*?([ABCD])", r"答案：.*?([ABCD])", "Answer:.*?([ABCD])", r"correct answer is.*?([ABCD])", r"\(([ABCD])\)"]
            for pattern in patterns:
                match = re.search(pattern, answer_output)
                if match:
                    answer_choice = match.group(1).upper()
                    break
        return answer_choice
    elif model_name.lower().startswith("olmo"):
        pattern = r"\(?([ABCD])\)?"
        match = re.search(pattern, answer_output.split("<|assistant|>")[-1])
        answer_choice = match.group(1).upper() if match else "X"
        return answer_choice
    elif model_name.lower().startswith("baichuan"):
        if len(answer_output) == 1:
            return answer_output
        else:
            pattern = f"\(?([ABCD])\)?"
            match = re.search(pattern, answer_output)
            answer_choice = match.group(1).upper() if match else "X"
            return answer_choice
    elif model_name.lower().startswith("meta-llama"):
        pattern = f"\(?([ABCD])\)?"
        match = re.search(pattern, answer_output)
        answer_choice = match.group(1).upper() if match else "X"
        return answer_choice
    elif model_name.lower().startswith("gemma"):
        answer_output = answer_output.replace("\n", "")
        pattern=r"###Answer:.*?([ABCD])"
        match = re.search(pattern, answer_output)
        answer_choice = match.group(1) if match else "X"
        return answer_choice
    elif model_name.lower().startswith("deepseek-r1"):
        answer_choice = "X"
        pattern = r"Answer:\s*\(?([ABCD])\)?"
        match = re.search(pattern, answer_output)
        if match:
            answer_choice = match.group(1)
        elif "</think>" in answer_output:
            answer_str = answer_output.split("</think>")[-1]
            pattern = r"\(([ABCD])\)"
            matches = re.findall(pattern, answer_str)
            if matches:
                answer_choice = matches[-1]
            elif "答案：" in answer_output:
                pattern = r"答案：\s*\(?([ABCD])\)?"
                match = re.search(pattern, answer_str)
                if match:
                    answer_choice = match.group(1)
        return answer_choice
    else:
        raise Exception(f"Unknown model {model_path}")

# lang_code="vie_Latn"
# model_name = "Meta-Llama-3-8B-Instruct_chat_template"
# answer_file = f"./bele_results/{lang_code}/{model_name}/answers.jsonl"
# num_X=0
# with jsonlines.open(answer_file) as reader:
#     for i, obj in enumerate(reader):
#         answer_choice = extract_answer_from_chat_templates(model_name, obj['answer'])
#         if answer_choice == "X":
#             num_X += 1
#         print(i, answer_choice)
#     print(num_X)
        

metrics = list()
for model_obj in models.to_dict(orient='records'):
    model_name = model_obj['model_name']
    is_instruction_tuned = model_obj['instruction_tuned']

    for lang in langs.to_dict(orient='records'):
        lang_code = lang['lang_code']
        lang_category = lang['lang_category']
        model_answers = list()
        if is_instruction_tuned:
            answer_file = f"./bele_results/{lang_code}/{model_name}_chat_template/answers.jsonl"
            extract_func = extract_answer_from_chat_templates
        else:
            answer_file = f"./bele_results/{lang_code}/{model_name}/answers.jsonl"
            extract_func = extract_answer_choice
        try:
            with jsonlines.open(answer_file) as reader:
                for obj in reader:
                    answer_choice = extract_func(model_name, obj['answer'])
                    model_answers.append(answer_choice.upper())
        except FileNotFoundError:
            print(model_name, lang_code, "FileNotFoundError")
            continue
        except InvalidLineError:
            print(model_name, lang_code, "InvalidLineError")
            continue
        except UnicodeDecodeError:
            print(model_name, lang_code, "UnicodeDecodeError")
            continue
        except AttributeError:
            print(model_name, lang_code, "AttributeError")
            continue
        else:
            if len(model_answers) != 900:
                print(model_name, lang_code, f"Only has {len(model_answers)} rows")
                continue
            else:
                print(model_name, lang_code)
        accuracy = sum(1 for x,y in zip(model_answers, correct_answers[lang_code]) if x == y) / len(correct_answers[lang_code])
        
        metrics.append({"model_name": model_name, 
                        "lang_category": lang_category,
                        "lang_code": lang_code, 
                        "lang_name": lang["lang_name"],
                        "accuracy": accuracy,
                        "answer_stats": Counter(model_answers)})

df = pd.DataFrame(metrics)
df = df.sort_values(["lang_category", "lang_code"])
df = df.join(models.set_index(["model_name"]), on='model_name', how='left')
df.to_csv("bele_output.csv", index=False)
