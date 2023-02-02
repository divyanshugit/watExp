import os
os.environ['CUDA_VISIBLE_DEVICES']="2,3,4"
os.environ["WANDB_DISABLED"] = "true"

import sys
# sys.path.append("/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/")
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import MBartForConditionalGeneration
from datasets import load_dataset,load_metric
from datasets import Dataset
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          EarlyStoppingCallback)

parser = argparse.ArgumentParser()
parser.add_argument('--s_lang', type=str, required=True, help="Enter the source language (Hindi, Kannada, Sanskrit)")
parser.add_argument('--t_lang', type=str, required=True, help="Enter the target language (Hindi, Kannada, Sanskrit")
args = parser.parse_args()

source_lang = args.s_lang.capitalize()
target_lang = args.t_lang.capitalize()

model_name = "ai4bharat/IndicBART"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)

model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path = model_name)

filepath = "/data1/home/piyushmishra/newgithub/parallel_data_output/enhikn.csv"

# dataset = load_dataset('csv', data_files=filepath, header=0, split='train')
df = pd.read_csv(filepath)
# df = pd.concat(map(pd.read_csv, ['/data1/home/piyushmishra/NLTM/NLTM-data/data_parallel/parallel_mkb_hsk_train_data.csv', '/data1/home/piyushmishra/NLTM/NLTM-data/data_parallel/parallel_mkb_hsk_test_data.csv']), ignore_index=True)

language_code =  {"Assamese": "as", "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn",
             "Malayalam": "ml", "Marathi": "mr", "Odia": "or", "Punjabi": "pa", "Tamil": "ta", "Telugu": "te", "English": "en"}

s_lang = language_code[source_lang.capitalize()]
t_lang = language_code[target_lang.capitalize()]

# os.environ["WANDB_PROJECT"] = s_lang+'-'+t_lang
# os.environ["WANDB_LOG_MODEL"] = "true" 

def converter(df, target_lang,s_lang,t_lang):
    def convert_devanagri(sentence):
        return UnicodeIndicTransliterator.transliterate(sentence, t_lang, s_lang)
    
    df['devanagari'] = df[target_lang].apply(lambda x: convert_devanagri(x))
    df.drop([target_lang],axis=1,inplace = True)
    df.rename(columns = {'devanagari':target_lang}, inplace = True)
    return df


df = converter(df,target_lang, s_lang,t_lang)
    
columns = [source_lang,target_lang]

df = pd.DataFrame(df,columns = columns)

dataset = Dataset.from_pandas(df,split='train')

dataset = dataset.train_test_split(test_size=0.10, shuffle=False)


print(dataset)



batch_size = 16
max_input_length = 128
max_target_length = 128




def preprocess_function(examples):
        inputs = [example + ' </s>' + f' <2{s_lang}>' for example in examples[source_lang]]
        targets = [f'<2{t_lang}> ' + example + ' </s>' for example in examples[target_lang]]

        model_inputs = tokenizer(inputs, max_length=max_input_length,padding=True, truncation=True)

        with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length,padding=True, truncation=True)

        #Changes
        labels['input_ids'] = [[(l if l!=tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']]

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

print(tokenized_datasets)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  

args = Seq2SeqTrainingArguments(
            s_lang+'-'+t_lang,
            #evaluation_strategy='epoch',
            evaluation_strategy='epoch',
            learning_rate=0.001,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.0001,
            logging_strategy='epoch',
        #     report_to='wandb',
            save_total_limit=2,
            num_train_epochs=50,
            save_strategy = "epoch",
            load_best_model_at_end=True,
            metric_for_best_model = 'bleu',
            predict_with_generate=True)


# from datasets import load_metric
metric = load_metric('sacrebleu')

def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

def compute_metrics(eval_preds):
        #print("compute_met called")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {'bleu': result['score']}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result['gen_len'] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        #print("compute-met-end")
        return result
early_stop = EarlyStoppingCallback(3)
trainer = Seq2SeqTrainer(
              model,
              args,
              train_dataset=tokenized_datasets['train'],
              eval_dataset=tokenized_datasets['test'],
              data_collator=data_collator,
              tokenizer=tokenizer,
              callbacks=[early_stop],
              compute_metrics=compute_metrics)  

trainer.train()
output_dir = f'/data1/home/piyushmishra/newgithub/NLTM/kd_exp/models/IndicBART_{source_lang}_{target_lang}2'
trainer.save_model(output_dir)

print(trainer.evaluate())
