# -*- coding: utf-8 -*-

import sys, os
import codecs
import json
from optparse import OptionParser

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from json_utils import load_json_file


def main():
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = OptionParser()
    parser.add_option("--lcovthres", dest="lcovthres", metavar="FLOAT", type="float", default=0.0,
                      help="eliminate languages with higher rate of missing values [0,1]")
    (options, args) = parser.parse_args()

    # input
    langlist_all = load_json_file(args[0])
    # output
    langfile = args[1]

    train_total = 0
    langlist = {}
    fsize = len(langlist_all[langlist_all.keys()[0]])
    for label, flist in langlist_all.iteritems():
        cov = 0
        for f,v in enumerate(flist):
            if v != -1:
                cov += 1
        if float(cov) / fsize >= options.lcovthres:
            langlist[label] = flist

    sys.stderr.write("language thresholding: %d -> %d\n" % (len(langlist_all), len(langlist)))

    with codecs.getwriter("utf-8")(open(langfile, 'w')) as f:
        f.write(json.dumps(langlist))

if __name__ == "__main__":
    main()
