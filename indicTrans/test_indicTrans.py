from datasets import load_metric, Dataset
import argparse


import pandas as pd
from engine import Model
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--s_lang",
    type=str,
    required=True,
    help="Enter the source language (Hindi, Kannada, Sanskrit)",
)

parser.add_argument(
    "--t_lang",
    type=str,
    required=True,
    help="Enter the source language (Hindi, Kannada, Sanskrit)",
)

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Enter the name of the dataset",
)

args = parser.parse_args()

source_lang = args.s_lang.capitalize()
target_lang = args.t_lang.capitalize()
dataset = args.dataset

bleu = load_metric("sacrebleu")

# df = pd.read_csv("/nlsasfs/home/ttbhashini/prathosh/divyanshu/IndicTrans/IndicTrans-MultilingualTranslation/M2M_prediction_Hindi_Kannada.csv")
# result = bleu.compute(predictions=df.prediction, references=df.Kannada)

# print(result)

# """
model = Model(
    expdir="/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/indicTrans/m2m/"
)
# """
print(model)

LANGUAGES = {
    "Bengali": "bn",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Odia": "or",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te",
}


# print("Source Lang: ",source_lang,"Target Lang: ",target_lang, "Lang Code", s_lang,t_lang)

if dataset == "FLORES":
    flores_lang = {
        "Bengali": "ben_Beng",
        "Gujarati": "guj_Gujr",
        "Hindi": "hin_Deva",
        "Kannada": "kan_Knda",
        "Malayalam": "mal_Mlym",
        "Marathi": "mar_Deva",
        "Odia": "ory_Orya",
        "Tamil": "tam_Taml",
        "Telugu": "tel_Telu",
    }
    try:
        s_lang = flores_lang[source_lang]
        t_lang = flores_lang[target_lang]

        path = "/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/flores/flores200_dataset/devtest"

        with open(
            f"{path}/{s_lang}.devtest", "r", encoding="utf-8", errors="ignore"
        ) as f:
            source_lang_data = f.readlines()

        with open(
            f"{path}/{t_lang}.devtest", "r", encoding="utf-8", errors="ignore"
        ) as f:
            target_lang_data = f.readlines()

    except:
        print("Language is not available in the corpus")

elif dataset == "wat_2021":
    LANGUAGES = {
        "Bengali": "bn",
        "Gujarati": "gu",
        "Hindi": "hi",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Marathi": "mr",
        "Odia": "or",
        "Punjabi": "pa",
        "Tamil": "ta",
        "Telugu": "te",
    }
    try:
        s_lang = LANGUAGES[source_lang]
        t_lang = LANGUAGES[target_lang]

        path = "/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/finalrepo/test"

        with open(f"{path}/test.{s_lang}", "r", encoding="utf-8", errors="ignore") as f:
            source_lang_data = f.readlines()
        with open(f"{path}/test.{t_lang}", "r", encoding="utf-8", errors="ignore") as f:
            target_lang_data = f.readlines()
    except:
        print("Language is not available in the corpus")

elif dataset == "NTREX":
    ntrex_lang = {
        "Bengali": "ben",
        "Gujarati": "guj",
        "Hindi": "hin",
        "Kannada": "kan",
        "Malayalam": "mal",
        "Marathi": "mar",
        "Tamil": "tam",
        "Telugu": "tel",
    }
    try:
        s_lang = ntrex_lang[source_lang]
        t_lang = ntrex_lang[target_lang]

        path = "/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/NTREX/NTREX-128"

        with open(
            f"{path}/newstest2019-ref.{s_lang}.txt",
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            source_lang_data = f.readlines()
        with open(
            f"{path}/newstest2019-ref.{t_lang}.txt",
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            target_lang_data = f.readlines()
    except:
        print("Language is not available in the corpus")


df = pd.DataFrame(
    {f"{source_lang}": source_lang_data, f"{target_lang}": target_lang_data}
)

df[f"{source_lang}"] = df[f"{source_lang}"].apply(lambda x: x.replace("\n", ""))
df[f"{target_lang}"] = df[f"{target_lang}"].apply(lambda x: x.replace("\n", ""))


print(df.head())


def get_predictions(df, source_lang, target_lang):
    print("Starting Predictions")

    def translate(text):
        return model.translate_paragraph(
            text, LANGUAGES[source_lang], LANGUAGES[target_lang]
        )
        # bleu.compute(prediction)

    df["predictions"] = df[f"{source_lang}"].apply(lambda x: translate(x))
    return df


print(df.columns)

df = get_predictions(df, source_lang=source_lang, target_lang=target_lang)

print(df.columns)

save_path = "/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/indicTrans/predictions"
df.to_csv(
    f"{save_path}/IndicTrans_pred{dataset}_{source_lang}_{target_lang}.csv", index=False
)


dataset = Dataset.from_pandas(df)

# Initialize an empty list to store the BLEU scores
scores = []

# Iterate through the dataset
for example in tqdm(dataset):
    ref = [example[f"{target_lang}"]]
    trans = [example["predictions"]]
    score = bleu.compute(predictions=trans, references=[ref])
    scores.append(score["score"])

print(f"BLEU score on {source_lang} to {target_lang}:", sum(scores) / len(scores) * 100)
