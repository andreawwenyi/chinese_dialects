import jsonlines
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
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

def extract_answer_choice(answer_output):
    try:
        answer_choice = answer_output.replace("\n", "").split("###Answer:")[-1].replace("#", "").replace('(','').replace(')','')[0]
    except:
        answer_choice = "X" # the model didn't provide answer
    return answer_choice

def extract_answer_from_chat_templates(model_name, answer_output):
    if model_name.lower().startswith("mistral"):
        pattern = r"\[/INST\]\s*\((\w)\)"
        match = re.search(pattern, answer_output)
        answer_choice = match.group(1).upper() if match else None
        return answer_choice
    elif (model_name.lower().startswith("yi")) or (model_name.lower().startswith("internlm")):
        if (len(answer_output) == 1):
            answer_choice = answer_output
        else:
            pattern = r"\((\w)\)"
            match = re.search(pattern, answer_output)
            answer_choice = match.group(1).upper() if match else None
        return answer_choice
    elif model_name.lower().startswith("olmo"):
        pattern = r"\(?(\w)\)?"
        match = re.search(pattern, answer_output.split("<|assistant|>")[-1])
        answer_choice = match.group(1).upper() if match else None
        return answer_choice
    elif model_name.lower().startswith("qwen"):
        pass

lang_code="khk_Cyrl"
model_name = "OLMo-2-1124-7B-Instruct_chat_template"
answer_file = f"./bele_results/{lang_code}/{model_name}/answers.jsonl"
with jsonlines.open(answer_file) as reader:
    for i, obj in enumerate(reader):
        answer_choice = extract_answer_from_chat_templates(model_name, obj['answer'])
        print(i, answer_choice)
        
# # one graph per model
# metrics = list()
# for model_name in models["model_name"]:
#     for lang in langs.to_dict(orient='records'):
#         lang_code = lang['lang_code']
#         lang_category = lang['lang_category']
#         answer_file = f"./bele_results/{lang_code}/{model_name}/answers.jsonl"
#         model_answers = list()
#         try:
#             with jsonlines.open(answer_file) as reader:
#                 for obj in reader:
#                     answer_choice = extract_answer_choice(obj['answer'])
#                     model_answers.append(answer_choice.upper())
#             accuracy = sum(1 for x,y in zip(model_answers, correct_answers[lang_code]) if x == y) / len(correct_answers[lang_code])
#         except FileNotFoundError:
#             print(model_name, lang_code, "FileNotFoundError")
#             continue
#         except InvalidLineError:
#             print(model_name, lang_code, "InvalidLineError")
#             continue
#         except UnicodeDecodeError:
#             print(model_name, lang_code, "UnicodeDecodeError")
#             continue
#         else:
            
#             if len(model_answers) != 900:
#                 print(model_name, lang_code, f"Only has {len(model_answers)} rows")
#             else:
#                 print(model_name, lang_code)
#         metrics.append({"model_name": model_name, 
#                         "lang_category": lang_category,
#                         "lang_code": lang_code, 
#                         "lang_name": lang["lang_name"],
#                         "accuracy": accuracy})
        
# df = pd.DataFrame(metrics)
# df = df.sort_values(["lang_category", "lang_code"])
# df = df.join(models.set_index(["model_name"]), on='model_name', how='left')
# df.to_csv("bele_output.csv", index=False)
