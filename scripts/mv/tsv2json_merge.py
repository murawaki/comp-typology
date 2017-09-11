# -*- coding: utf-8 -*-
import sys, os
import codecs
import glob
import json
from collections import Counter

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from json_utils import load_json_file, load_json_stream

def main(orig, src, fpath, dst):
    fid2struct = load_json_file(fpath)
    langs = list(load_json_stream(open(orig)))
    for lang in langs:
        lang["counted_features"] = [Counter() for feature in fid2struct]
        lang["features_filled"] = {}
        for fnode in fid2struct:
            lang["features_filled"][fnode["wals_id"]] = -1

    for fpath in glob.glob(src + ".*"):
        sys.stderr.write("processing {}\n".format(fpath))
        with open(fpath) as fin:
            fin.readline() # ignore the header
            for lang, l in zip(langs, fin):
                l = l.rstrip()
                a = l.split("\t")
                label = a.pop(0)
                for fid, v in enumerate(a):
                    lang["counted_features"][fid][int(v)] += 1

    for lang in langs:
        binsize = 0
        xfreq = Counter()
        for fid, (fnode, counts) in enumerate(zip(fid2struct, lang["counted_features"])):
            size = len(fnode["vid2label"])
            wals_id = fnode["wals_id"]
            maxv, maxvv = -1, -1
            for i in xrange(size):
                xfreq[binsize+i] += counts[i]
                # if lang["xfreq"][binsize+i] >= maxvv:
                if counts[i] >= maxvv:
                    maxvv = counts[i]
                    maxv = i
                lang["features_filled"][wals_id] = maxv
            binsize += size
        del lang["counted_features"]
        lang["xfreq"] = [xfreq[i] for i in xrange(binsize)]

    with codecs.getwriter("utf-8")(open(dst, 'w')) as fout:
        for lang in langs:
            fout.write("%s\n" % json.dumps(lang))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
