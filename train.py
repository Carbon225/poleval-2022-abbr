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
from torch.utils.data import ConcatDataset
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
from poleval_dataset import load_poleval_dataset, load_kw_dataset
from mixed_dataset import MixedDataset

NUM_PROC = min(mp.cpu_count(),16)


def make_run_name():
    now = datetime.datetime.now()
    return f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}-{os.environ.get("SLURM_JOB_ID", "local")}'


@dataclass
class ModelArguments:
    checkpoint: str = field(default='allegro/plt5-base')


@dataclass
class DataArguments:
    dataset: str = field(default='poleval')
    # values:
    #   - mixed
    #   - poleval
    #   - poleval-dev
    #   - wiki
    #   - kw
    #   - poleval+kw
    #   - poleval-dev+kw

    wiki_mixed_weight: int = field(default=4)
    poleval_mixed_weight: int = field(default=1)


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=64)
    gradient_accumulation_steps: int = field(default=4)

    learning_rate: float = field(default=0.000015)
    weight_decay: float = field(default=0.0001)
    warmup_ratio: float = field(default=0.1)
    num_train_epochs: int = field(default=200)
    max_steps: int = field(default=-1) # -1 to use num_train_epochs
    early_stopping_patience: int = field(default=-1) # -1 to disable

    save_strategy: str = field(default='steps')
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=5)

    logging_strategy: str = field(default='steps')
    logging_steps: int = field(default=10)

    evaluation_strategy: str = field(default='steps')
    eval_steps: int = field(default=100)

    metric_for_best_model: str = field(default='eval_aw')
    greater_is_better: bool = field(default=True)
    load_best_model_at_end: bool = field(default=True)

    output_dir: str = field(default=f'results/{make_run_name()}')
    report_to: str = field(default='wandb')
    optim: str = field(default='adamw_torch')
    predict_with_generate: bool = field(default=True)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)

    generation_max_length: int = field(default=100)

    seed: int = field(default=42)
    lr_scheduler_type: str = field(default='linear')

def train():
    parser = HfArgumentParser((MyTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses(args_file_flag='--from_file')

    print('Loading model')
    tokenizer = AutoTokenizer.from_pretrained(model_args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.checkpoint)

    print('Loading metrics')
    exact_match = evaluate.load('exact_match')

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        split_preds = [pred.split(';') for pred in decoded_preds]
        split_labels = [label.split(';') for label in decoded_labels]

        #FIXME: support ; in predictions, but much better change the separator

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
    kw_dataset = load_kw_dataset('data/kw_uniq.tsv')

    print('Tokenizing')
    def tokenize(batch):
        if 'labels' in batch:
            return tokenizer(text=batch['text'], text_target=batch['labels'])
        else:
            return tokenizer(text=batch['text'])

    if data_args.dataset in ('wiki', 'mixed'):
        wiki_dataset = wiki_dataset.map(tokenize, batched=True, num_proc=NUM_PROC, remove_columns=['text']).shuffle(seed=42)
    # DO NOT SHUFFLE TEST
    poleval_dataset = poleval_dataset.map(tokenize, batched=True, num_proc=NUM_PROC, remove_columns=['text'])
    kw_dataset = kw_dataset.map(tokenize, batched=True, num_proc=NUM_PROC, remove_columns=['text'])

    if data_args.dataset == 'mixed':
        train_dataset = MixedDataset(
            wiki_dataset,
            poleval_dataset['train'],
            weights=[
                data_args.wiki_mixed_weight,
                data_args.poleval_mixed_weight,
            ],
        )
    elif data_args.dataset == 'poleval':
        train_dataset = poleval_dataset['train']
    elif data_args.dataset == 'poleval-dev':
        train_dataset = ConcatDataset([
            poleval_dataset['train'],
            poleval_dataset['dev-0'],
        ])
    elif data_args.dataset == 'wiki':
        train_dataset = wiki_dataset
    elif data_args.dataset == 'kw':
        train_dataset = kw_dataset
    elif data_args.dataset == 'poleval+kw':
        train_dataset = ConcatDataset([
            kw_dataset,
            poleval_dataset['train'],
        ])
    elif data_args.dataset == 'poleval-dev+kw':
        train_dataset = ConcatDataset([
            kw_dataset,
            poleval_dataset['train'],
            poleval_dataset['dev-0'],
        ])

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=poleval_dataset['dev-0'],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
            ),
        ] if training_args.early_stopping_patience > 0 else [],
    )

    if training_args.do_train:
        print('Training')
        trainer.train()
        trainer.save_model(os.path.join(training_args.output_dir, 'final'))

    if training_args.do_eval:
        print('Evaluating')
        trainer.evaluate()
        for ds_name in ('test-A', 'test-B'):
            trainer.evaluate(poleval_dataset[ds_name], metric_key_prefix=f'eval_{ds_name}')

    if training_args.do_predict:
        print('Predicting')
        for ds_name in ('dev-0', 'test-A', 'test-B'):
            preds = trainer.predict(poleval_dataset[ds_name]).predictions
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            with open(os.path.join(training_args.output_dir, f'{ds_name}-preds.tsv'), 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                for pred in decoded_preds:
                    split_pred = pred.split(';')
                    if len(split_pred) == 4:
                        full_pred = '; '.join(split_pred[0:2]).strip()
                        base_pred = '; '.join(split_pred[2:4]).strip()
                        writer.writerow((full_pred, base_pred))
                    else:
                        full_pred = split_pred[0].strip()
                        base_pred = split_pred[1].strip() if len(split_pred) > 1 else ''
                        writer.writerow((full_pred, base_pred))


if __name__ == '__main__':
    train()
