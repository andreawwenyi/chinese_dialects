import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

langs = pd.read_csv("langs.csv", header=None)
langs.columns = ["lang_code", "lang_name", "lang_category"]
models = pd.read_csv("models.csv", header=None)
models.columns = ["model_type", "model_path"]
models["model_name"] = models["model_path"].apply(lambda t: t.split("/")[-1])

# one graph per model

df = pd.DataFrame()
for model_name in models["model_name"]:
    for lang in langs.to_dict(orient="records"):
        log_file = f"./results/dev_{lang['lang_code']}/{model_name}/logs.txt"
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
df['log(ppl)'] = np.log(df['ppl'])
df = df.sort_values("lang_category")
g = sns.catplot(df, x="lang_code", y='log(ppl)', hue='lang_category', kind="bar", row="model_name", height=3, aspect=3)
g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
g.set_titles(row_template="{row_name}")
g.tight_layout()
plt.savefig("ppl_bar.png", dpi=300)

df["tok_efficiency"] = df["n_toks"] / df["n_chars"]
g = sns.catplot(df, x="lang_code", y='tok_efficiency', hue='lang_category', kind="bar", row="model_name", height=3, aspect=3)
g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
g.set_titles(row_template="{row_name}")
g.tight_layout()
plt.savefig("tok_bar.png", dpi=300)
