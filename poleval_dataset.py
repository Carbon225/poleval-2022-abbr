import pandas as pd
import datasets
import os
import re

IGNORED = re.compile(r'[^a-zA-Zęóąśłżźćń0-9.!?</>]+')


def sanitize(text: str):
    text = IGNORED.sub(' ', text)
    return text.strip()


def load_kw_dataset(path: str):
    train_data = []
    for line in open(path):
        line = line.strip('\n')
        input, output = line.split('\t')
        input = input.replace('<abbrev>', '<mask>')
        input = input.replace('</abbrev>', '</mask>')
        output = output.replace(' <sep> ', '; ')
        input = ' ' + input
        output = ' ' + output
        train_data.append([input, output])

    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text", "labels"]
    
    return datasets.Dataset.from_pandas(train_df).shuffle(42)


def load_poleval_dataset():
    root = os.path.join(os.path.dirname(__file__), '2022-abbreviation-disambiguation')

    header_in = pd.read_csv(os.path.join(root, 'in-header.tsv'), sep='\t').columns.values.tolist()
    header_expected = pd.read_csv(os.path.join(root, 'out-header.tsv'), sep='\t').columns.values.tolist()
    header_in, header_expected


    df_train_in = pd.read_csv(os.path.join(root, 'train', 'in.tsv'),
        sep='\t',
        names=header_in,
        quoting=3,
    )
    df_train_expected = pd.read_csv(os.path.join(root, 'train', 'expected.tsv'),
        sep='\t',
        names=header_expected,
        quoting=3,
    )
    df_train = pd.concat([df_train_in, df_train_expected], axis=1)
    df_train = df_train.dropna()


    df_dev_in = pd.read_csv(os.path.join(root, 'dev-0', 'in.tsv'),
        sep='\t',
        names=header_in,
        quoting=3,
    )
    df_dev_expected = pd.read_csv(os.path.join(root, 'dev-0', 'expected.tsv'),
        sep='\t',
        names=header_expected,
        quoting=3,
    )
    df_dev = pd.concat([df_dev_in, df_dev_expected], axis=1)
    df_dev = df_dev.dropna()

    df_test_a = pd.read_csv(os.path.join(root, '..', 'poleval-test', 'test-A.csv'))
    df_test_b = pd.read_csv(os.path.join(root, '..', 'poleval-test', 'test-B.csv'))

    dataset = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(df_train).shuffle(42),
        'dev-0': datasets.Dataset.from_pandas(df_dev),#.shuffle(42),
        'test-A': datasets.Dataset.from_pandas(df_test_a),
        'test-B': datasets.Dataset.from_pandas(df_test_b),
    })

    def f(example):
        if 'FullForm' not in example:
            return {
                'text': ' ' + example['Context'].replace('<mask>', f'<mask>{example["Abbrev"]}</mask>', 1),
            }
        else:
            return {
                # 'text': ' ' + sanitize(example['Context'].replace('<mask>', f'<mask>{example["Abbrev"]}</mask>', 1)),
                'text': ' ' + example['Context'].replace('<mask>', f'<mask>{example["Abbrev"]}</mask>', 1),
                'labels': ' ' + example['FullForm'] + '; ' + example['BaseForm']
            }

    dataset = dataset.map(f, batched=False)

    for split in dataset.keys():
        dataset[split] = dataset[split].remove_columns(
            [name for name in dataset[split].column_names if name not in ('text', 'labels')]
        )

    return dataset


if __name__ == '__main__':
    print(load_poleval_dataset())
