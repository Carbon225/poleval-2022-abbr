from datasets import load_dataset
import re
from random import choice, randint
import multiprocessing as mp
import spacy

from poleval_dataset import SEPARATOR

NUM_PROC = mp.cpu_count()

VALID_TEXT = re.compile(r'(?: [a-zęóąśłżźćń]{5}[a-zęóąśłżźćń]*)+ ', re.IGNORECASE | re.UNICODE)
WHITESPACE = re.compile(r'\s+')

nlp = spacy.load('pl_core_news_lg')


def abbr_first(t):
    n = randint(1, 4)
    return t[:n] + '.'


def abbr_first_last(t):
    return t[0] + t[-1]


def abbr_first_mid(t):
    i = randint(2, len(t) - 3)
    return t[0] + t[i] + '.'


def abbr_first_mid_last(t):
    i = randint(2, len(t) - 3)
    return t[0] + t[i] + t[-1]


STRATEGIES = [abbr_first, abbr_first_last, abbr_first_mid, abbr_first_mid_last]


def make_abbr(t):
    strategy = choice(STRATEGIES)
    return strategy(t)


BATCH_SIZE_MIN, BATCH_SIZE_MAX = (140, 200)


def transform_batch(examples):
    out = {
        'text': [],
        'labels': [],
    }

    for t in examples['text']:
        i = 0
        while i < len(t) - BATCH_SIZE_MIN + 1:
            batch_size = randint(BATCH_SIZE_MIN, BATCH_SIZE_MAX)
            text = t[i:i+batch_size]
            i += batch_size

            text = WHITESPACE.sub(' ', text)

            matches = VALID_TEXT.findall(text)
            if len(matches) == 0: continue

            fragment = choice(matches).strip()
            words = fragment.split(' ')

            words = words[:randint(1, len(words))]
            fragment = ' '.join(words)

            words_abbr = [make_abbr(w) for w in words]
            fragment_abbr = ' '.join(words_abbr)

            text_abbr = text.replace(' ' + fragment + ' ', f' <mask>{fragment_abbr}</mask> ', 1)

            doc = nlp(fragment)
            fragment_lemma = ' '.join([t.lemma_ for t in doc])

            out['text'].append(text_abbr)
            out['labels'].append(' ' + fragment + SEPARATOR + fragment_lemma)

    return out


def load_wiki_dataset():
    ds = load_dataset('olm/wikipedia', language='pl', date='20221101', split='train')
    ds_abbr = ds.map(
        transform_batch,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
    )
    return ds_abbr


if __name__ == '__main__':
    ds = load_wiki_dataset()
    ds.save_to_disk('wiki_dataset')
