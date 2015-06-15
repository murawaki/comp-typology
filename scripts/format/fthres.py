# -*- coding: utf-8 -*-

import sys, os
import codecs
import numpy as np
import json
from optparse import OptionParser

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from json_utils import load_json_file


def shrink_features(leaf_list, fid2struct, fcovthres):
    fcount = np.zeros(len(fid2struct), dtype=np.int32)
    lcount = np.zeros(len(leaf_list), dtype=np.int32)
    for l, node in enumerate(leaf_list):
        for f,v in enumerate(node["featureList"]):
            if v != -1:
                fcount[f] += 1
                lcount[l] += 1

    old_coverage = fcount.sum()
    new2old = []
    fid2struct2 = []
    for i,v in enumerate(fcount):
        if float(v) / len(leaf_list) >= fcovthres:
            new2old.append(i)
            fid2struct2.append(fid2struct[i])

    sys.stderr.write("shrinking features: %d -> %d (reduction: %f)\n" % \
                         (len(fid2struct),
                          len(new2old),
                          (1.0 - len(new2old) / float(len(fid2struct))) ))
    new2old = np.array(new2old, dtype=np.int32)

    new_coverage = 0
    for node in leaf_list:
        flist = -1 * np.ones(len(new2old), dtype=np.int32)
        for nid, oid in enumerate(new2old):
            if node["featureList"][oid] != -1:
                flist[nid] = node["featureList"][oid]
                new_coverage += 1
        node["featureList"] = flist


    total1 = float(len(leaf_list) * len(fid2struct))
    total2 = float(len(leaf_list) * len(new2old))
    sys.stderr.write("coverage %f -> %f\n" % \
                         (old_coverage / total1,
                          new_coverage / total2))
    return fid2struct2, new2old


def main():
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = OptionParser()
    parser.add_option("--fcovthres", dest="fcovthres", metavar="FLOAT", type="float", default=0.0,
                      help="eliminate features with higher rate of missing values [0,1]")
    parser.add_option("--feature2", dest="feature2", metavar="FILE", default=None)
    (options, args) = parser.parse_args()

    # input
    tree = load_json_file(args[0])
    fid2struct = load_json_file(args[1])
    # output
    langfile = args[2]
    fidmapfile = args[3]

    leaf_list = []
    queue = [(tree, None)]
    while len(queue) > 0:
        node, pnode = queue.pop(0)
        if "children" in node:
            for cname, cnode in node["children"].iteritems():
                queue.append((cnode, node))
        else:
            leaf_list.append(node)

    if options.fcovthres > 0.0:
        fid2struct, fidmap = shrink_features(leaf_list, fid2struct, options.fcovthres)
        np.save(fidmapfile, fidmap)

    if options.feature2 is not None:
        with codecs.getwriter("utf-8")(open(options.feature2, 'w')) as f:
            f.write(json.dumps(fid2struct))

    train_total = 0
    langlist = {}
    fsize = len(leaf_list[0]["featureList"])
    for node in leaf_list:
        langlist[node["label"]] = node["featureList"].tolist()

    with codecs.getwriter("utf-8")(open(langfile, 'w')) as f:
        f.write(json.dumps(langlist))


if __name__ == "__main__":
    main()
