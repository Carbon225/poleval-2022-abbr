import os
os.environ["WANDB_ENTITY"]="carbon-agh"
os.environ["WANDB_PROJECT"]="poleval-2022-abbr"
os.environ["WANDB_LOG_MODEL"]="false"
os.environ["WANDB_WATCH"]="false"

from dataclasses import dataclass, field
import csv
import datetime
import multiprocessing as mp
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
)
import evaluate
from datasets import load_from_disk
from poleval_dataset import load_poleval_dataset
from mixed_dataset import MixedDataset


NUM_PROC = mp.cpu_count()


def make_run_name():
    now = datetime.datetime.now()
    return f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}'


@dataclass
class ModelArguments:
    checkpoint: str = field(default='allegro/plt5-base')
    compile: bool = field(default=False)


@dataclass
class DataArguments:
    dataset: str = field(default='poleval')
    test_split: str = field(default='dev-0')
    wiki_mixed_weight: int = field(default=4)
    poleval_mixed_weight: int = field(default=1)


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=64)
    gradient_accumulation_steps: int = field(default=4)

    # from JK final submission
    learning_rate: float = field(default=0.000015)
    weight_decay: float = field(default=0.0001)
    warmup_ratio: float = field(default=0.1)

    max_steps: int = field(default=20000)
    early_stopping_patience: int = field(default=20)

    save_strategy: str = field(default='steps')
    save_steps: int = field(default=50)
    save_total_limit: int = field(default=5)

    logging_strategy: str = field(default='steps')
    logging_steps: int = field(default=5)

    evaluation_strategy: str = field(default='steps')
    eval_steps: int = field(default=50)

    metric_for_best_model: str = field(default='eval_aw')
    greater_is_better: bool = field(default=True)
    load_best_model_at_end: bool = field(default=True)

    output_dir: str = field(default=f'results/{make_run_name()}')
    report_to: str = field(default='wandb')
    optim: str = field(default='adamw_torch')
    predict_with_generate: bool = field(default=True)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)


def train():
    parser = HfArgumentParser((MyTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    print('Loading model')
    tokenizer = AutoTokenizer.from_pretrained(model_args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.checkpoint)
    if model_args.compile:
        model = torch.compile(model)

    print('Loading metrics')
    exact_match = evaluate.load('exact_match')

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        split_preds = [pred.split(';') for pred in decoded_preds]
        split_labels = [label.split(';') for label in decoded_labels]

        full_preds = [pred[0].strip() for pred in split_preds]
        base_preds = [pred[1].strip() if len(pred) > 1 else '' for pred in split_preds]

        full_labels = [label[0].strip() for label in split_labels]
        base_labels = [label[1].strip() for label in split_labels]

        af = exact_match.compute(predictions=full_preds, references=full_labels, ignore_case=True)['exact_match']
        ab = exact_match.compute(predictions=base_preds, references=base_labels, ignore_case=True)['exact_match']
        aw = 0.25 * af + 0.75 * ab

        return {
            'af': af,
            'ab': ab,
            'aw': aw,
        }

    print('Loading datasets')
    if data_args.dataset in ('wiki', 'mixed'):
        wiki_dataset = load_from_disk('wiki_dataset')
    poleval_dataset = load_poleval_dataset()

    print('Tokenizing')
    def tokenize(batch):
        if 'labels' in batch:
            return tokenizer(text=batch['text'], text_target=batch['labels'])
        else:
            return tokenizer(text=batch['text'])

    if data_args.dataset in ('wiki', 'mixed'):
        wiki_dataset = wiki_dataset.map(tokenize, batched=True, num_proc=NUM_PROC, remove_columns=['text']).shuffle(seed=42)
    # DO NOT SHUFFLE TEST
    poleval_dataset = poleval_dataset.map(tokenize, batched=True, num_proc=NUM_PROC, remove_columns=['text'])#.shuffle(seed=42)

    if data_args.dataset == 'mixed':
        train_dataset = MixedDataset(
            wiki_dataset,
            poleval_dataset['train'],
            weights=[
                data_args.wiki_mixed_weight,
                data_args.poleval_mixed_weight,
            ],
        )

    if data_args.dataset == 'poleval':
        train_dataset = poleval_dataset['train']

    if data_args.dataset == 'wiki':
        train_dataset = wiki_dataset

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=poleval_dataset[data_args.test_split],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
            ),
        ],
    )

    if training_args.do_train:
        print('Training')
        try:
            trainer.train()
        except KeyboardInterrupt:
            print('Interrupted')
    
    if training_args.do_eval:
        print('Evaluating')
        trainer.evaluate()

    if training_args.do_predict:
        print('Predicting')
        for ds_name in ('dev-0', 'test-A', 'test-B'):
            predictions = trainer.predict(poleval_dataset[ds_name])
            decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
            with open(f'{ds_name}-preds.tsv', 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                for pred in decoded_preds:
                    split_pred = pred.split(';')
                    full_pred = split_pred[0].strip()
                    base_pred = split_pred[1].strip() if len(split_pred) > 1 else ''
                    writer.writerow((full_pred, base_pred))


if __name__ == '__main__':
    train()
