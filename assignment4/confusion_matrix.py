import sys

def confusion_matrix(filename):
    f = open(filename, 'r')
    ll, lm, lo, lp, ml, mm, mo, mp, ol, om, oo, op, pl, pm, po, pp = 0
    for line in f.readlines():
        cols = line.split('\t')
        word = cols[0]
        label = cols[1]
        pred = cols[2]
        if 'LOC' in label:
            if 'LOC' in pred:
                ll += 1
            elif 'MISC' in pred:
                lm += 1
            elif 'ORG' in pred:
                lo += 1
            elif 'PER' in pred:
                lp += 1
        elif 'MISC' in label:
            if 'LOC' in pred:
                ml += 1
            elif 'MISC' in pred:
                mm += 1
            elif 'ORG' in pred:
                mo += 1
            elif 'PER' in pred:
                mp += 1
        elif 'ORG' in pred:
            if 'LOC' in pred:
                ol += 1
            elif 'MISC' in pred:
                om += 1
            elif 'ORG' in pred:
                oo += 1
            elif 'PER' in pred:
                op += 1
        elif 'PER' in pred:
            if 'LOC' in pred:
                pl += 1
            elif 'MISC' in pred:
                pm += 1
            elif 'ORG' in pred:
                po += 1
            elif 'PER' in pred:
                pp += 1

    print('confusion matrix')
    print('----------------------------------------------------')
    print('           LOC        MISC       ORG          PER')
    print('LOC         %d         %d         %d          %d' % (ll, lm, lo, lp))
    print('MISC        %d         %d         %d          %d' % (ml, mm, mo, mp))
    print('ORG         %d         %d         %d          %d' % (ol, om, oo, op))
    print('PER         %d         %d         %d          %d' % (pl, pm, po, pp))



def main():
    filename = sys.argv[1]
    print(confusion_matrix(filename))


if __name__ == '__main__':
    main()