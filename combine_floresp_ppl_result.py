import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.errors import ParserError
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
        log_file = f"./floresp_results_v2/dev_{lang['lang_code_flores']}/{model_name}/logs_2.txt"
        try:
            log = pd.read_csv(log_file, sep="\t")
        except FileNotFoundError:
            print(model_name, lang["lang_code_flores"], "FileNotFoundError")
            continue
        except ParserError:
            print(model_name, lang["lang_code_flores"], "ParserError")
            continue
        else:
            if log.shape[0] != 997:
                print(model_name, lang["lang_code_flores"], "Incomplete")
                continue
            elif 'nll_sum' not in log.columns:
                print(model_name, lang["lang_code_flores"], "Missing nll_sum column")
            elif log[log['nll_sum'].isnull()].shape[0] > 0:
                print(model_name, lang["lang_code_flores"], "Has Null Values")
                continue
            else:
                print(model_name, lang["lang_code_flores"])
        log = log.reset_index()
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

# calculate IP
for base_lang in ['eng_Latn', 'cmn_Hans']:
    base_lang_result = df[df['lang_code'] == base_lang].copy()
    base_lang_result = base_lang_result.pivot(index='index', columns='model_name', values='nll_sum')
    df[f'ip_base_{base_lang}'] = None

    output = pd.DataFrame()
    for model_name in df['model_name'].unique():
        for lang_code in df['lang_code'].unique():
            model_lang_df = df[(df['model_name'] == model_name) & (df['lang_code'] == lang_code)].copy()
            model_lang_df = model_lang_df.join(base_lang_result, on='index')
            
            ip = model_lang_df['nll_sum'] / model_lang_df[model_name]
            df.loc[(df['model_name'] == model_name) & (df['lang_code'] == lang_code), f'ip_base_{base_lang}'] = ip

    df.to_csv("floresp_nll_output_v2.csv", index=False)

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