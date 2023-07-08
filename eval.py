import sys

path_gold = sys.argv[1]
path_pred = sys.argv[2]

f_gold = open(path_gold).readlines()
f_pred = open(path_pred).readlines()

af = ab = a = 0

for i, (gold, pred) in enumerate(zip(f_gold, f_pred)):
    gold = gold.strip().split('\t')
    # pred=pred.replace('#','\t')
    pred = pred.strip().split('\t')



    try:
        gold_form, gold_base = gold
        pred_form, pred_base = pred
    except:
        print(gold, pred)

    # if True:
    #     if ';' in pred_form:
    #         pred_form, pred_base = pred_form.split(';')

    a += 1
    if gold_form.strip().lower() == pred_form.strip().lower():
        af += 1
    else:
        print('AF', i, [gold_form, pred_form], file=sys.stderr)
    if gold_base.strip().lower() == pred_base.strip().lower():
        ab += 1
    else:
        print('AB', i, [gold_base, pred_base], file=sys.stderr)
print(f'{(0.25 * af + 0.75 * ab) / a*100:.2f}')
