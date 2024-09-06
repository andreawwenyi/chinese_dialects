import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

langs = pd.read_csv("langs.csv")
models = pd.read_csv("models.csv", header=None)
models.columns = ["model_type", "model_path"]
models["model_name"] = models["model_path"].apply(lambda t: t.split("/")[-1])
model_order = models["model_name"]
# one graph per model

df = pd.DataFrame()
for model_name in models["model_name"]:
    for lang in langs.to_dict(orient="records"):
        log_file = f"./results/dev_{lang['lang_code_flores']}/{model_name}/logs_2.txt"
        try:
            log = pd.read_csv(log_file, sep="\t")
        except:
            continue
        if len(log) != 997:
            continue
        log["lang_code"] = lang["lang_code_flores"]
        log["lang_name"] = lang["lang_name"]
        log["lang_category"] = lang["lang_category"]
        log["model_name"] = model_name

        with open(f"./data/floresp-v2.0-rc.2/dev/dev.{lang['lang_code_flores']}", "r") as f:
            sents = f.readlines()
        n_chars = [len(s) for s in sents]
        log["n_chars"] = n_chars
        df = pd.concat((df, log))
df = df.sort_values(["lang_category", "lang_code"])
df['nll_sum'] = np.log(df['ppl']) * df['n_toks']
df.to_csv("nll_output.csv", index=False)

# lang_order=langs['lang_code'].values
# df_mean = df.groupby(["model_name", "lang_category", "lang_code"])["nll_sum"].mean().reset_index(name="nll_sum")
# g = sns.catplot(data=df_mean, y="lang_code", x='nll_sum', jitter=False, hue="model_name", 
#                 order=lang_order,
#                 # row='lang_category', sharey=False, 
#                 aspect=1.62
#                 )
# plt.savefig("nll_sum_boxplot.png", dpi=300)

# g = sns.catplot(df, x="lang_code", y='nll_sum', hue='lang_category', kind="bar", 
#                 row="model_name", row_order=model_order,
#                 height=3, aspect=3)
# g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
# g.set_titles(row_template="{row_name}")
# g.tight_layout()
# plt.savefig("nll_sum_bar.png", dpi=300)

df["tok_efficiency"] = df["n_toks"] / df["n_chars"]
# df = df[~df['lang_code'].isin(["yue_Hant", "uig_Arab"])]
g = sns.catplot(df, x="lang_code", y='tok_efficiency', hue='lang_category', kind="bar", 
                col="model_name", col_wrap=2, col_order = model_order,
                height=2, aspect=3)
g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
g.set_titles(col_template="{col_name}")
g.set_ylabels("fertility")
g.tight_layout()
plt.savefig("./figures/tok_bar.pdf")
