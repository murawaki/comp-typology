# -*- coding: utf-8 -*-
#
# filter out features by language coverage
#
import sys, os
import codecs
from collections import defaultdict
import json
from argparse import ArgumentParser

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from json_utils import load_json_file, load_json_stream


def main():
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = ArgumentParser()
    parser.add_argument("--fthres", dest="fthres", metavar="FLOAT", type=float, default=0.0,
                        help="eliminate features with higher rate of missing values [0,1]")
    parser.add_argument("langs_in", metavar="LANGS_IN", default=None)
    parser.add_argument("flist_in", metavar="FLIST_IN", default=None)
    parser.add_argument("langs_out", metavar="LANGS_OUT", default=None)
    parser.add_argument("flist_out", metavar="FLIST_OUT", default=None)
    args = parser.parse_args()

    bad_features = [
        "39B", # Inclusive/Exclusive Forms in Pama-Nyungan -- only covers Pama-Nyungan
        "81B", # Languages with two Dominant Orders of Subject, Object, and Verb -- no value for non-applicable
        "139A", # Irregular Negatives in Sign Languages
        "140A", # Question Particles in Sign Languages"
        "141A", # Writing Systems -- irrelevant
    ]

    fid2struct = load_json_file(args.flist_in)
    langs =[lang for lang in load_json_stream(open(args.langs_in))]

    lang_total = len(langs)
    fsize_in = len(fid2struct)
    fcounts = defaultdict(int)
    for lang in langs:
        for k, v in lang["features"].iteritems():
            fcounts[k] += 1

    fsurvived = 0
    fid2struct2 = []
    wals_id_list = {}
    for i, struct in enumerate(fid2struct):
        wid = struct["wals_id"]
        freq = fcounts[wid]
        if wid not in bad_features and float(freq) / lang_total >= args.fthres:
            struct["fid"] = len(fid2struct2)
            fid2struct2.append(struct)
            wals_id_list[wid] = True
            fsurvived += 1

    deleted = 0
    for lang in langs:
        for k in lang["features"].keys():
            if k not in wals_id_list:
                del lang["features"][k]
                deleted += 1

    sys.stderr.write("shrink features: %d -> %d (%f%%)\n" % (fsize_in, fsurvived, fsurvived / float(fsize_in)))
    sys.stderr.write("%d elements deleted\n" % deleted)

    with codecs.getwriter("utf-8")(open(args.flist_out, 'w')) as f:
        f.write("%s\n" % json.dumps(fid2struct2))

    with codecs.getwriter("utf-8")(open(args.langs_out, 'w')) as f:
        for lang in langs:
            f.write("%s\n" % json.dumps(lang))



if __name__ == "__main__":
    main()
