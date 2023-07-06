import collections
import random
from argparse import ArgumentParser


def read_data(path):
    for in_line in open(path):
        if in_line[-1] == '\n': in_line = in_line[:-1]
        tokens = in_line.split('\t')
        yield tokens


def voting1(labels):
    if len(set(labels)) == 1:
        return labels[0]

    s = collections.defaultdict(int)
    for l in labels:
        s[l] += 1

    sl = sorted(s.items(), key=lambda x: x[1], reverse=True)

    if sl[0][1] > sl[1][1]:
        return sl[0][0]

    return random.choice([sl[0][0], sl[1][0]])


if __name__ == "__main__":
    parser = ArgumentParser(description='')
    parser.add_argument('in_path', nargs='+', help='paths to in.tsv')
    args = parser.parse_args()

    for file_labels in zip(*[read_data(path) for path in args.in_path]):
        outputs1 = []
        outputs2 = []
        for output1, output2 in file_labels:
            outputs1.append(output1)
            outputs2.append(output2)

        new_label1 = voting1(outputs1)
        new_label2 = voting1(outputs2)

        print('\t'.join([new_label1, new_label2]))
