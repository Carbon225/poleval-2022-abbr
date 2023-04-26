from torch.utils.data import IterableDataset
import random


class MixedDataset(IterableDataset):
    def __init__(self, *datasets, weights=None):
        self.datasets = datasets
        self.weights = weights if weights is not None else [1] * len(datasets)

    def __iter__(self):
        i = [0] * len(self.datasets)
        ds = [d for d in range(len(self.datasets)) for _ in range(self.weights[d])]
        random.shuffle(ds)
        d = 0
        while True:
            di = ds[d]
            yield self.datasets[di][i[di]]
            i[di] = (i[di] + 1) % len(self.datasets[di])
            d = (d + 1) % len(ds)
