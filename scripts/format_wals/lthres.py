# -*- coding: utf-8 -*-

import sys, os
import codecs
import json
from argparse import ArgumentParser

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from json_utils import load_json_file


def main():
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = ArgumentParser()
    parser.add_argument("--lcovthres", dest="lcovthres", metavar="FLOAT", type=float, default=0.0,
                        help="eliminate languages with higher rate of missing values [0,1]")
    parser.add_argument("langs_all", metavar="INPUT", default=None)
    parser.add_argument("langs", metavar="OUTPUT", default=None)
    args = parser.parse_args()

    # input
    langlist_all = load_json_file(args.langs_all)
    # output
    langfile = args.langs

    train_total = 0
    langlist = {}
    fsize = len(langlist_all[langlist_all.keys()[0]])
    for label, flist in langlist_all.iteritems():
        cov = 0
        for f,v in enumerate(flist):
            if v != -1:
                cov += 1
        if float(cov) / fsize >= args.lcovthres:
            langlist[label] = flist

    sys.stderr.write("language thresholding: %d -> %d\n" % (len(langlist_all), len(langlist)))

    with codecs.getwriter("utf-8")(open(langfile, 'w')) as f:
        f.write(json.dumps(langlist))

if __name__ == "__main__":
    main()
