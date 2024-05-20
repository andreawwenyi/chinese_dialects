import jsonlines
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from jsonlines.jsonlines import InvalidLineError

langs = pd.read_csv("bele_langs.csv", header=None)
langs.columns = ["lang_code", "lang_name", "lang_category"]
models = pd.read_csv("models.csv", header=None)
models.columns = ["model_type", "model_path"]
models["model_name"] = models["model_path"].apply(lambda t: t.split("/")[-1])
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

# one graph per model
metrics = list()
for model_name in models["model_name"]:
    for lang in langs.to_dict(orient='records'):
        lang_code = lang['lang_code']
        lang_category = lang['lang_category']
        answer_file = f"./bele_results/{lang_code}/{model_name}/answers.jsonl"
        model_answers = list()
        try:
            with jsonlines.open(answer_file) as reader:
                for obj in reader:
                    answer_choice = extract_answer_choice(obj['answer'])
                    model_answers.append(answer_choice.upper())
            accuracy = sum(1 for x,y in zip(model_answers, correct_answers[lang_code]) if x == y) / len(correct_answers[lang_code])
        except FileNotFoundError:
            print(model_name, lang_code, "FileNotFoundError")
            continue
        except InvalidLineError:
            print(model_name, lang_code, "InvalidLineError")
            continue
        else:
            print(model_name, lang_code)
            if len(model_answers) != 900:
                continue
        metrics.append({"model_name": model_name, 
                        "lang_category": lang_category,
                        "lang_code": lang_code, 
                        "accuracy": accuracy})
df = pd.DataFrame(metrics)
df = df.sort_values(["lang_category", "lang_code"])
g = sns.catplot(df, x="lang_code", y='accuracy', 
                hue='lang_category', 
                kind="bar", 
                row="model_name", 
                row_order=model_order,
                height=3, aspect=3)

g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
g.set_titles(row_template="{row_name}")
g.tight_layout()
plt.savefig("bele_accuracy.png", dpi=300)
