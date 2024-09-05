"""
Transliterate Kazakh in Cyrillic script to Arabic script. 
"""
import jsonlines
from utils import kazakh_convert_cyrl_to_arab

# flores
input_file="./data/floresp-v2.0-rc.2/dev/dev.kaz_Cyrl"
output_file="./data/floresp-v2.0-rc.2/dev/dev.kaz_Arab"
arab_sentences = list()
with open(input_file, "r") as f: 
    sentences = f.readlines()
    for sent in sentences:
        transliteration = kazakh_convert_cyrl_to_arab.translate(sent, 'kk-arab')
        arab_sentences.append(transliteration)

with open(output_file, "w") as f:
    for sent in arab_sentences:
        f.write(sent)


# bele
input_file = "data/Belebele/kaz_Cyrl.jsonl"
output_file = "data/Belebele/kaz_Arab.jsonl"
with jsonlines.open(output_file, "w") as writer:
    with jsonlines.open(input_file, "r") as reader:
        for obj in reader:
            transliteration = dict()
            for key in obj.keys():
                if key in ["flores_passage", "question", 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4']:
                    transliteration[key] = kazakh_convert_cyrl_to_arab.translate(obj[key], 'kk-arab')
                elif key == "dialect":
                    transliteration[key] = "kaz_Arab"
                else:
                    transliteration[key] = obj[key]
            writer.write(transliteration)

