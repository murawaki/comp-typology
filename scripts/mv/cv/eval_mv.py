# -*- coding: utf-8 -*-

import sys, os
import codecs
import json
import numpy as np
import random 
from collections import defaultdict
from argparse import ArgumentParser

sys.path.insert(1, os.path.join(os.path.join(sys.path[0], os.path.pardir), os.path.pardir))
from json_utils import load_json_file, load_json_stream

def eval_mv(filledlist, langs):
    total = 0
    correct = 0
    for lang, flang in zip(langs, filledlist):
        for wals_id, v in lang["features"].iteritems():
            if wals_id not in flang["features"]:
                total += 1
                if flang["features_filled"][wals_id] == v:
                    correct += 1
    return (total, correct)

def eval_most_frequent(fid2struct, hidelist, langs):
    # NOTE: most frequent values are based on hidelist, not langs (reference)
    fsize = len(fid2struct)
    wals_id2histogram = {}
    for hlang in hidelist:
        for wals_id, v in hlang["features"].iteritems():
            if wals_id not in wals_id2histogram:
                wals_id2histogram[wals_id] = defaultdict(int)
            wals_id2histogram[wals_id][v] += 1
    wals_id2maxk = {}
    for wals_id, histogram in wals_id2histogram.iteritems():
        maxk, maxv = None, -1
        for k, v in histogram.iteritems():
            if v >= maxv:
                maxv = v
                maxk = k
        wals_id2maxk[wals_id] = maxk

    total = 0
    correct = 0
    for lang, hlang in zip(langs, hidelist):
        for wals_id, v in lang["features"].iteritems():
            if wals_id not in hlang["features"]:
                total += 1
                if wals_id2maxk[wals_id] == v:
                    correct += 1
    return (total, correct)

def eval_random(fid2struct, langs):
    total = 0
    correct = 0

    wals_id2size = {}
    for struct in fid2struct:
        wals_id2size[struct["wals_id"]] = len(struct["vid2label"])
    for lang in langs:
        for wals_id, v in lang["features"].iteritems():
            total += 1
            r = np.random.random_integers(0, wals_id2size[struct["wals_id"]] - 1)
            if v == r:
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

    langs = [lang for lang in load_json_stream(open(args.langs))]
    if args.random:
        fid2struct = load_json_file(args.f2)
        total, correct = eval_random(fid2struct, langs)
    elif args.most_frequent:
        hidelist = [lang for lang in load_json_stream(open(args.f1))]
        fid2struct = load_json_file(args.f2)
        total, correct = eval_most_frequent(fid2struct, hidelist, langs)
    else:
        filledlist = [lang for lang in load_json_stream(open(args.f1))]
        total, correct = eval_mv(filledlist, langs)
    sys.stdout.write("%f\t%d\t%d\n" % (float(correct) / total, correct, total))


    
if __name__ == "__main__":
    main()
