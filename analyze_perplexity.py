import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

langs = pd.read_csv("floresp_langs.csv", header=None)
langs.columns = ["lang_code", "lang_name", "lang_category"]
models = pd.read_csv("models.csv", header=None)
models.columns = ["model_type", "model_path"]
models["model_name"] = models["model_path"].apply(lambda t: t.split("/")[-1])
model_order = models["model_name"]
# one graph per model

df = pd.DataFrame()
for model_name in models["model_name"]:
    for lang in langs.to_dict(orient="records"):
        # log_file = f"./results/dev_{lang['lang_code']}/{model_name}/logs.txt"
        log_file = f"./results/dev_{lang['lang_code']}/{model_name}/logs_2.txt"
        try:
            log = pd.read_csv(log_file, sep="\t")
        except:
            continue
        if len(log) != 997:
            continue
        log["lang_code"] = lang["lang_code"]
        log["lang_name"] = lang["lang_name"]
        log["lang_category"] = lang["lang_category"]
        log["model_name"] = model_name

        with open(f"./data/floresp-v2.0-rc.2/dev/dev.{lang['lang_code']}", "r") as f:
            sents = f.readlines()
        n_chars = [len(s) for s in sents]
        log["n_chars"] = n_chars
        df = pd.concat((df, log))
df = df.sort_values(["lang_category", "lang_code"])

df['nll_sum'] = np.log(df['ppl']) * df['n_toks']
g = sns.catplot(df, x="lang_code", y='nll_sum', hue='lang_category', kind="bar", 
                row="model_name", row_order=model_order,
                height=3, aspect=3)
g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
g.set_titles(row_template="{row_name}")
g.tight_layout()
plt.savefig("nll_sum_bar.png", dpi=300)

df["nll_char"] = df['nll_sum'] / df['n_chars']
print(df.columns)
g = sns.catplot(df, x="lang_code", y='nll_char', hue='lang_category', kind="bar", 
                row="model_name", row_order=model_order,
                height=3, aspect=3)
g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
g.set_titles(row_template="{row_name}")
g.tight_layout()
plt.savefig("nll_char_bar.png", dpi=300)
print(df.head())

df["tok_efficiency"] = df["n_toks"] / df["n_chars"]
g = sns.catplot(df, x="lang_code", y='tok_efficiency', hue='lang_category', kind="bar", 
                row="model_name", row_order = model_order,
                height=3, aspect=3)
g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
g.set_titles(row_template="{row_name}")
g.tight_layout()
plt.savefig("tok_bar.png", dpi=300)
