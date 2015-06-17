# -*- coding: utf-8 -*-

import sys, os
import codecs
import json
import numpy as np
import random 
from collections import defaultdict
from argparse import ArgumentParser

sys.path.insert(1, os.path.join(os.path.join(sys.path[0], os.path.pardir), os.path.pardir))
from json_utils import load_json_file

def eval_mv(filledlist, hidelist, langlist):
    total = 0
    correct = 0
    for label, flist in langlist.iteritems():
        for fid, rv in enumerate(flist):
            if rv >= 0 and hidelist[label][fid] == -1:
                total += 1
                if filledlist[label][fid] == rv:
                    correct += 1
    return (total, correct)

def eval_most_frequent(hidelist, langlist):
    # NOTE: most frequent values are based on hidelist, not langlist (reference)
    fsize = len(langlist[langlist.keys()[0]])
    fid_freq_list = [defaultdict(int) for _ in xrange(fsize)]
    for label, flist in hidelist.iteritems():
        for fid, v in enumerate(flist):
            if v >= 0:
                fid_freq_list[fid][v] += 1
    fid_maxv = [-1 for _ in xrange(fsize)]
    for fid in xrange(fsize):
        _sorted = sorted(fid_freq_list[fid].keys(), key=lambda x: fid_freq_list[fid][x], reverse=True)
        fid_maxv[fid] = _sorted[0]

    total = 0
    correct = 0
    for label, flist in langlist.iteritems():
        for fid, rv in enumerate(flist):
            if rv >= 0 and hidelist[label][fid] == -1:
                total += 1
                if fid_maxv[fid] == rv:
                    correct += 1
    return (total, correct)

def eval_random(fid2struct, langlist):
    total = 0
    correct = 0
    for label, flist in langlist.iteritems():
        for fid, rv in enumerate(flist):
            if rv >= 0:
                total += 1
                r = np.random.random_integers(0, len(fid2struct[fid]["vid2label"]) - 1)
                if rv == r:
                    correct += 1
    return (total, correct)


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", metavar="INT", type=int, default=None,
                        help="random seed")
    parser.add_argument("--random", dest="random", action="store_true", default=False)
    parser.add_argument("--freq", dest="most_frequent", action="store_true", default=False)
    parser.add_argument("langs", metavar="LANGS", default=None)
    parser.add_argument("f1", metavar="DUMMY_OR_LANGS_FILLED_OR_LANGS_HIDDEN", default=None)
    parser.add_argument("f2", metavar="FID2STRUCT_OR_DUMMY_OR_LANGS_HIDDEN", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    langlist = load_json_file(args.langs)
    if args.random:
        fid2struct = load_json_file(args.f2)
        total, correct = eval_random(fid2struct, langlist)
    elif args.most_frequent:
        hidelist = load_json_file(args.f1)
        total, correct = eval_most_frequent(hidelist, langlist)
    else:
        filledlist = load_json_file(args.f1)
        hidelist = load_json_file(args.f2)
        total, correct = eval_mv(filledlist, hidelist, langlist)
    sys.stdout.write("%f\t%d\t%d\n" % (float(correct) / total, correct, total))


    
if __name__ == "__main__":
    main()
