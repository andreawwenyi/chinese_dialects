# chinese_dialects

## Environment
```
conda create -n {env_name} python=3.11
conda activate {env_name}
pip install -r requirements.txt
```
## Files
- langs.csv
- models.csv
- floresp_models.csv --> used by jobs/calc_ppl_on_floresp.sh for a list of languages to run. 
## Scripts

### Language zero-shot multiple choice MRC on [belebele](https://github.com/facebookresearch/belebele) dataset

1. Run zero-shot inference
```sh
python3 multiple_choice.py --model-name ${model_name} --output-dir ${output_dir} --lang-id ${lang}
```
`lang-id`: 
`model_name`: huggingface model name, e.g. `mistralai/Mistral-7B-v0.3`

2. Combine zero-shot results
```sh
python3 combine_bele_result.py
```
will save output to `bele_output.csv`.

### Language model's perplexity on the [mc2](https://github.com/luciusssss/mc2_corpus) corpus
1. Calculate perplexity
```sh
python3 calc_ppl_on_mc2.py --model-name ${model_name} --output-dir ${output_dir} --lang ${lang}
```
`lang`: one of `tibetan`, `kazakh`, `mongolian`, `uyghur`
`model_name`: huggingface model name, e.g. `mistralai/Mistral-7B-v0.3`

2. Combine perplexity files
```sh
python3 combine_mc2_ppl_result.py
```

3. Make figures
```sh

```

### Calculate language model's perplexity on the [Flores+](https://huggingface.co/datasets/openlanguagedata/flores_plus) corpus
1. Calculate perplexity
```sh
python3 calc_ppl_on_floresp.py --model-name ${model_name} --output-dir ${output_dir} --lang ${lang}
```
`lang`: Flores+ {language code}_{script}. e.g. `ace_Arab`, `eng_Latn`. 
`model_name`: huggingface model name, e.g. `mistralai/Mistral-7B-v0.3`

2. Combine perplexity files 
```sh
python3 combine_floresp_ppl_result.py
```
will save output to `floresp_nll_output.csv`.


### Make figures
```sh
Rscript plot_box_plots.R
```

