import pandas as pd
import argparse
import numpy as np

from datasets import Dataset, load_metric

from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s_lang",
        type=str,
        required=True,
        help="Enter the any Indic Language present in Samanatar Dataset",
    )
    parser.add_argument(
        "--t_lang",
        type=str,
        required=True,
        help="Enter the any Indic Language present in Samanatar Dataset",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset(wat_2021, flores, ntrex)"
    )
    args = parser.parse_args()

    source_language = args.s_lang
    target_language = args.t_lang

    LANGUAGE_KEY = {
            "Assamese": "as",
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
    if args.dataset == "wat_2021":
        

        s_lang = LANGUAGE_KEY[source_language]
        t_lang = LANGUAGE_KEY[target_language]
        
        path = f"/home/ece/divyanshu/finalrepo/test"
        # firstSetSI, firstSetSII, secondSetSI, secondSetSII subject to change with a better variable name

        with open(f"{path}/test.{s_lang}", "r", encoding="utf-8", errors="ignore") as f:
            firstSetS = f.readlines()

        with open(
            f"{path}/test.{t_lang}", "r", encoding="utf-8", errors="ignore"
        ) as f:
            secondSetS = f.readlines()

    elif args.dataset == "ntrex":

        LANGUAGES = {
            "Bengali": "ben",
            "Gujarati": "guj",
            "Hindi": "hin",
            "Kannada": "kan",
            "Malayalam": "mal",
            "Marathi": "mar",
            "Tamil": "tam",
            "Telugu": "tel",           
        }

        s_lang = LANGUAGES[source_language]
        t_lang = LANGUAGES[target_language]
        # newstest2019-ref.arb.txt
        path = "/home/ece/divyanshu/NTREX/NTREX-128"
        with open(f"{path}/newstest2019-ref.{s_lang}.txt", "r", encoding="utf-8", errors="ignore") as f:
            firstSetS = f.readlines()

        with open(
            f"{path}/newstest2019-ref.{t_lang}.txt", "r", encoding="utf-8", errors="ignore"
        ) as f:
            secondSetS = f.readlines()

    elif args.dataset == "flores":
        LANGUAGES = {
            "Assamese": "asm_Beng",
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
        
        s_lang = LANGUAGES[source_language]
        t_lang = LANGUAGES[target_language]

        #ace_Arab.devtest
        path = "/home/ece/divyanshu/flores200_dataset/devtest"
        with open(f"{path}/{s_lang}.devtest", "r", encoding="utf-8", errors="ignore") as f:
            firstSetS = f.readlines()

        with open(
            f"{path}/{t_lang}.devtest", "r", encoding="utf-8", errors="ignore"
        ) as f:
            secondSetS = f.readlines()


    print(f"firstSetSI: {len(firstSetS)}, Unique firstSetSI: {len(set(firstSetS))}")
    print(f"secondSetSI: {len(secondSetS)}, Unique secondSetSI: {len(set(secondSetS))}")

    df = pd.DataFrame(
        {f"{source_language}": firstSetS, f"{target_language}": secondSetS}
    )

    df[f"{source_language}"] = df[f"{source_language}"].apply(
        lambda x: x.replace("\n", "")
    )
    df[f"{target_language}"] = df[f"{target_language}"].apply(
        lambda x: x.replace("\n", "")
    )

    def converter(df, language, lang_code):
        def convert_devanagari(sentence):
            return UnicodeIndicTransliterator.transliterate(sentence, lang_code, "hi")

        original_sentences = df[language]
        df["devanagari"] = df[language].apply(lambda x: convert_devanagari(x))
        df.drop([language], axis=1, inplace=True)
        print("Done Devanagari Conversion")
        df.rename(columns={"devanagari": language}, inplace=True)

        return df, original_sentences

    if source_language != "Hindi":
        df, source_original_sentences = converter(df, source_language, s_lang)
    else:
        df, target_original_sentences = converter(df, target_language, t_lang)

    dataset = Dataset.from_pandas(df)

    model_name = f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/models/{source_language}{target_language}"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, do_lower_case=False, use_fast=False, keep_accents=True
    )
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    max_input_length = 256
    max_target_length = 256

    metric = load_metric("sacrebleu")

    def preprocess_function(examples):
        inputs = [
            example + " </s>" + f" <2{s_lang}>" for example in examples[source_language]
        ]
        targets = [
            f"<2{t_lang}> " + example + " </s>" for example in examples[target_language]
        ]

        model_inputs = tokenizer(
            inputs, max_length=max_input_length, padding=True, truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding=True, truncation=True
            )

        # Changes
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    batch_size = 16

    args = Seq2SeqTrainingArguments(
        "translation",
        evaluation_strategy="epoch",
        do_train=False,
        do_predict=True,
        learning_rate=0.001,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.0001,
        predict_with_generate=True,
    )

    # Changes
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    trainer = Seq2SeqTrainer(
        model,
        args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    predict_dataset = tokenized_datasets
    predict_results = trainer.predict(
        predict_dataset, metric_key_prefix="predict", max_length=max_target_length
    )

    print(f"The prediction metric:{predict_results}")

    predictions = tokenizer.batch_decode(
        predict_results.predictions,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    predictions = [pred.strip() for pred in predictions]

    def get_original_sentences(predictions, source_lang, target_lang):
        predicted_sentences = []
        for sentence in predictions:
            sentence = UnicodeIndicTransliterator.transliterate(
                sentence, "hi", {t_lang}
            )
            predicted_sentences.append(sentence)

        predictions_data = pd.DataFrame(
            {
                f"{source_language}": source_lang,
                f"{target_language}": target_lang,
                "predictions": predicted_sentences,
            }
        )
        return predictions_data

    if source_language != "Hindi":
        prediction_data = get_original_sentences(
            predictions,
            source_lang=source_original_sentences,
            target_lang=tokenized_datasets[target_language],
        )
    else:
        prediction_data = get_original_sentences(
            predictions,
            source_lang=tokenized_datasets[source_language],
            target_lang=target_original_sentences,
        )

    prediction_data.to_csv(f"{source_language}_{target_language}_wat2021.csv")


if __name__ == "__main__":
    main()
