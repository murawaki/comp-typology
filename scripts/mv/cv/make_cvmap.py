# -*- coding: utf-8 -*-
#
import sys, os
import codecs
import json
import random
from optparse import OptionParser

sys.path.insert(1, os.path.join(os.path.join(sys.path[0], os.path.pardir), os.path.pardir))
from json_utils import load_json_file


def main():
    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", metavar="INT", type="int", default=None,
                      help="random seed")
    (options, args) = parser.parse_args()

    if options.seed is not None:
        random.seed(options.seed)

    src, dst = args[0], args[1]
    cvn = int(args[2])
    langlist = load_json_file(src)

    filled_list = []
    for label, flist in langlist.iteritems():
        for fid, v in enumerate(flist):
            if v >= 0:
                filled_list.append((label, fid))

    random.shuffle(filled_list)

    # N-fold cross-validation
    cell_size = len(filled_list) / cvn
    cell_size2 = len(filled_list) % cvn

    cvmap = [[] for i in xrange(cvn)]
    for i in xrange(cvn):
        cell_start = cell_size * i + min(i, cell_size2)
        cell_len = cell_size + (i < cell_size2)
        for j in xrange(cell_start, cell_start + cell_len):
            cvmap[i].append(filled_list[j])

    with codecs.getwriter("utf-8")(open(dst, 'w')) as f:
        f.write(json.dumps(cvmap))


if __name__ == "__main__":
    main()
