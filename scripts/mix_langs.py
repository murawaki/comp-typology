# -*- coding: utf-8 -*-
#
# create mixtures of two languages
#
import sys
import codecs
import numpy as np
from argparse import ArgumentParser

from rand_utils import rand_partition
from json_utils import load_json_file, load_json_stream
from evaluator import CategoricalFeatureListEvaluator

def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="src", metavar="WALS_CODE", default="khm")
    parser.add_argument("--dst", dest="dst", metavar="WALS_CODE", default="khr")
    parser.add_argument("--ssize", dest="ssize", metavar="INT", type=int, default=100)
    parser.add_argument("--samples", dest="samples", metavar="INT", type=int, default=1000)
    parser.add_argument("--cat", dest="is_hid", action="store_false", default=True,
                        help="mixtures of categorical features (using Bernoulli distribution)")
    parser.add_argument("--hid", dest="is_hid", action="store_true",
                        help="mixtures of hidden features (linear interpolation)")
    parser.add_argument("--auto", dest="is_auto", action="store_true", default=False,
                        help="adjust hidden features")
    parser.add_argument("model", metavar="MODEL", default=None)
    parser.add_argument("langs", metavar="LANGS", default=None)
    args = parser.parse_args()

    with codecs.getreader("utf-8")(open(args.model)) as reader:
        dat = reader.read()
        evaluator = CategoricalFeatureListEvaluator.loads(dat)
    fid2struct = evaluator.fid2struct
    wals_code2lang = {}
    for lang in load_json_stream(open(args.langs)):
        wals_code2lang[lang["wals_code"]] = lang

    # khm: Khmer
    # khr: Munda
    k = np.array(wals_code2lang[args.src]["catvect_filled"], dtype=np.int32)
    m = np.array(wals_code2lang[args.dst]["catvect_filled"], dtype=np.int32)

    if args.is_hid:
        kh = evaluator.encode(evaluator.cat2bin(k))
        mh = evaluator.encode(evaluator.cat2bin(m))
        for i in xrange(args.ssize + 1):
            r = float(i) / args.ssize
            hv = (1.0 - r) * kh + r * mh
            if args.is_auto:
                bin2 = evaluator.binarize(evaluator.decode(hv))
                cat2 = evaluator.bin2cat(bin2)
                # sys.stderr.write("%f\t%s\n" % (r, fid2struct[50]["vid2label"][cat2[50]]))
                hv = evaluator.encode(bin2)
            score = evaluator.calc_score(hv)
            sys.stdout.write("%f\t%f\n" % (r, score))
    else:
        for i in xrange(args.ssize + 1):
            r = float(i) / args.ssize
            ts = np.zeros(args.samples)
            for t in xrange(args.samples):
                fv = np.zeros(evaluator.catsize, dtype=np.int32)
                for fi in xrange(evaluator.catsize):
                    iv = rand_partition([1.0-r, r])
                    if iv == 0:
                        fv[fi] = k[fi]
                    else:
                        fv[fi] = m[fi]
                score = evaluator.calc_score(evaluator.encode(evaluator.cat2bin(fv)))
                ts[t] = score
            avg = np.mean(ts)
            std = np.std(ts)
            sys.stdout.write("%f\t%f\t%f\n" % (r, avg, std))

if __name__ == "__main__":
    main()
