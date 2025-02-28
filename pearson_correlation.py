import pandas as pd
import numpy as np
import random
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("floresp_nll_output_v2.csv")
df = df[df['model_source'].isin(["China", "US/Europe"])].copy()
df = df[df['lang_name'] != "English"].copy()

df = df.groupby(['lang_code', 'lang_name', 'lang_category', "model_name", "model_name_abrev", "model_source"])["ip"].mean().reset_index(name='mean_ip')
df = df.groupby(['lang_code', 'lang_name', 'lang_category', "model_source"])["mean_ip"].mean().reset_index(name='mean_ip')

df = df.pivot(index=['lang_name', 'lang_category'], columns='model_source', values='mean_ip').reset_index()
corrcoef, p_val = stats.pearsonr(df['China'], df['US/Europe'])
print("Pearson Corr: ", corrcoef, "P value: ", p_val)
df_no_cmn_hans = df[df['lang_name'] != "Mandarin (Simplified)"]
print("Pearson Corr w/o Simplified Chinese: ", stats.pearsonr(df_no_cmn_hans['China'], df_no_cmn_hans['US/Europe']))

sns.scatterplot(data = df, x='US/Europe', y='China', hue='lang_category')


for i, lang_name in enumerate(df['lang_name']):
    if lang_name == 'Mandarin (Simplified)':
        annotate_x_offset = -0.05
        annotate_y_offset = -0.025
        plt.annotate(lang_name, (df['US/Europe'].iloc[i]+annotate_x_offset, df['China'].iloc[i] + annotate_y_offset), size=8)
    elif lang_name == "Yue (Cantonese)":
        annotate_x_offset = -0.1
        annotate_y_offset = random.uniform(0.01, 0.02)
        plt.annotate(lang_name, (df['US/Europe'].iloc[i]+annotate_x_offset, df['China'].iloc[i] + annotate_y_offset), size=8)
    elif lang_name in ['Mandarin (Traditional)']:
        annotate_x_offset = -0.1
        annotate_y_offset = random.uniform(0.01, 0.02)
        plt.annotate(lang_name, (df['US/Europe'].iloc[i]+annotate_x_offset, df['China'].iloc[i] + annotate_y_offset), size=8)

# Add regression line
z = np.polyfit(df['US/Europe'], df['China'], 1)
p = np.poly1d(z)
plt.plot(df['US/Europe'], p(df['US/Europe']), color='darkgrey', linestyle='--', alpha=0.8)
plt.xlabel('US/Europe models average IP')
plt.ylabel('China models average IP')
plt.title(f'US/Europe vs China model Performance (r = {corrcoef:.3f}, p value = {p_val:.3f})')
plt.grid(True, alpha=0.3)
plt.xlim(0.15, 0.92)
plt.ylim(0.15, 0.92)
plt.legend(title='Language Category', bbox_to_anchor=(0.97, 0.3), borderaxespad=0, fontsize=7)
plt.tight_layout()
plt.savefig("figures/floresp_ip_scatter.pdf")
plt.close()

df = pd.read_csv("bele_output.csv")
df = df[df['model_source'].isin(["China", "US/Europe"])].copy()

df = df.groupby(['lang_code', 'lang_name', 'lang_category', "model_source"])["accuracy"].mean().reset_index(name='mean_accuracy')
df = df.pivot(index=['lang_name', 'lang_category'], columns='model_source', values='mean_accuracy').reset_index()
corrcoef, p_val = stats.pearsonr(df['China'], df['US/Europe'])
print("Pearson Corr: ", corrcoef, "P value: ", p_val)
df_no_cmn_hans = df[df['lang_name'] != "Mandarin (Simplified)"]
print("Pearson Corr w/o Simplified Chinese: ", stats.pearsonr(df_no_cmn_hans['China'], df_no_cmn_hans['US/Europe']))

sns.scatterplot(data = df, x='US/Europe', y='China', hue='lang_category')

for i, lang_name in enumerate(df['lang_name']):
    if lang_name in 'Mandarin (Simplified)':
        annotate_x_offset = -0.03
        annotate_y_offset = random.uniform(0.01, 0.02)
        plt.annotate(lang_name, (df['US/Europe'].iloc[i]+annotate_x_offset, df['China'].iloc[i] + annotate_y_offset), size=8)
    elif lang_name == 'Mandarin (Traditional)':
        annotate_x_offset = -0.15
        annotate_y_offset = -0.03
        plt.annotate(lang_name, (df['US/Europe'].iloc[i]+annotate_x_offset, df['China'].iloc[i] + annotate_y_offset), size=8)

# Add regression line
z = np.polyfit(df['US/Europe'], df['China'], 1)
p = np.poly1d(z)
plt.plot(df['US/Europe'], p(df['US/Europe']), color='darkgrey', linestyle='--', alpha=0.8)
plt.xlabel('US/Europe models average accuracy')
plt.ylabel('China models average accuracy')
plt.title(f'US/Europe vs China model Performance (r = {corrcoef:.3f}, p value = {p_val:.3f})')
plt.grid(True, alpha=0.3)
plt.xlim(0.15, 0.92)
plt.ylim(0.15, 0.92)
plt.legend(title='Language Category', bbox_to_anchor=(0.97, 0.3), borderaxespad=0, fontsize=7)
plt.tight_layout()
plt.savefig("figures/bele_accuracy_scatter.pdf")