import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

langs = pd.read_csv("langs.csv")
langs = langs[langs["lang_code_flores"].notnull()]
models = pd.read_csv("models.csv", header=None)
models.columns = ["model_name_abrev", "model_path_hf", "model_source"]
models["model_name"] = models["model_path_hf"].apply(lambda t: t.split("/")[-1])
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
df = df.join(models.set_index(["model_name"]), on='model_name', how='left')
df['nll_sum'] = np.log(df['ppl']) * df['n_toks']
df.to_csv("floresp_nll_output.csv", index=False)

df["tok_efficiency"] = df["n_toks"] / df["n_chars"]
# df = df[~df['lang_code'].isin(["yue_Hant", "uig_Arab"])]
g = sns.catplot(df, x="lang_code", y='tok_efficiency', hue='lang_category', kind="bar", 
                col="model_name", col_wrap=2, col_order = model_order,
                height=2, aspect=3)
g.tick_params(axis='x', which='both', rotation=40, labelsize=10)
g.set_titles(col_template="{col_name}")
g.set_ylabels("fertility")
g.tight_layout()
plt.savefig("./figures/floresp_tok_bar.pdf")


# langs = ["kaz_Arab", "kaz_Cyrl"]
# models = pd.read_csv("models.csv", header=None)
# models.columns = ["model_type", "model_path"]
# models["model_name"] = models["model_path"].apply(lambda t: t.split("/")[-1])
# model_order = models["model_name"]
# one graph per model

# df = pd.DataFrame()
# for model_name in models["model_name"]:
#     for lang in langs:
#         log_file = f"./results/dev_{lang}/{model_name}/logs_2.txt"
#         try:
#             log = pd.read_csv(log_file, sep="\t")
#         except:
#             continue
#         if len(log) != 997:
#             continue
#         log["lang_code"] = lang
#         log["model_name"] = model_name

#         with open(f"./data/floresp-v2.0-rc.2/dev/dev.{lang}", "r") as f:
#             sents = f.readlines()
#         n_chars = [len(s) for s in sents]
#         log["n_chars"] = n_chars
#         df = pd.concat((df, log))
# df['nll_sum'] = np.log(df['ppl']) * df['n_toks']

# df["tok_efficiency"] = df["n_toks"] / df["n_chars"]
# # df = df[~df['lang_code'].isin(["yue_Hant", "uig_Arab"])]
# g = sns.catplot(df, x="lang_code", y='tok_efficiency', kind="bar", 
#                 col="model_name", col_wrap=2, col_order = model_order,
#                 height=2, aspect=3)
# g.tick_params(axis='x', which='both', labelsize=10)
# g.set_titles(col_template="{col_name}")
# g.set_ylabels("fertility")
# g.tight_layout()
# plt.savefig("./figures/kaz_tok_bar.pdf")

