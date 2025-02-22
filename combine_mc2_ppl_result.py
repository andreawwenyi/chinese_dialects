import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
langs = [
    {"lang_code": "tibetan", "lang_name": "Tibetan"},
    {"lang_code": "kazakh", "lang_name": "Kazakh"},
    {"lang_code": "mongolian", "lang_name": "Mongolian"},
    {"lang_code": "uyghur", "lang_name": "Uyghur"},
]

models = pd.read_csv("models.csv", header=None)
models.columns = ["model_name_abrev", "model_path_hf", "model_source"]
models["model_name"] = models["model_path_hf"].apply(lambda t: t.split("/")[-1])
model_order = models["model_name"]
# one graph per model

df = pd.DataFrame()
for model_name in models["model_name"]:
    for lang in langs:
        log_file = f"./results/mc2/{lang['lang_code']}/{model_name}/logs_2.txt"
        try:
            log = pd.read_csv(log_file, sep="\t")
        except:
            continue
        print(model_name)
        print(lang)
        print(len(log))
        log["lang_code"] = lang["lang_code"]
        log["lang_name"] = lang["lang_name"]
        log["lang_category"] = "Chinese Ethnic Minorities"
        log["model_name"] = model_name
        df = pd.concat((df, log))
df = df.sort_values(["lang_category", "lang_code"])
df = df.join(models.set_index(["model_name"]), on='model_name', how='left')
df['nll_sum'] = np.log(df['ppl']) * df['n_toks']
df.to_csv("mc2_nll_output.csv", index=False)

# g = sns.catplot(df, x="lang_code", y='tok_efficiency', hue='lang_category', kind="bar", 
#                 col="model_name", col_wrap=2, col_order = model_order,
#                 height=2, aspect=3)
# g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
# g.set_titles(col_template="{col_name}")
# g.set_ylabels("fertility")
# g.tight_layout()
# plt.savefig("./figures/mc2_tok_bar.pdf")