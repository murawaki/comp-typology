# -*- coding: utf-8 -*-
#
import sys, os
import codecs
import json
import random
from argparse import ArgumentParser

sys.path.insert(1, os.path.join(os.path.join(sys.path[0], os.path.pardir), os.path.pardir))
from json_utils import load_json_file, load_json_stream


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", metavar="INT", type=int, default=None,
                        help="random seed")
    parser.add_argument("src", metavar="SOURCE", default=None)
    parser.add_argument("flist", metavar="FLIST", default=None)
    parser.add_argument("dst", metavar="DESTINATION", default=None)
    parser.add_argument("cvn", metavar="INT", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    src, dst = args.src, args.dst
    cvn = int(args.cvn)
    langs = [lang for lang in load_json_stream(open(src))]
    fid2struct = load_json_file(args.flist)

    filled_list = []
    for lang in langs:
        for wals_id, v in lang["features"].iteritems():
            filled_list.append((lang["wals_code"], wals_id))
        # for struct in fid2struct:
        #     wals_id = struct["wals_id"]
        #     if wals_id in lang["features"]:
        #         fid = struct["fid"]
        #         filled_list.append((lang["wals_code"], fid))

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
