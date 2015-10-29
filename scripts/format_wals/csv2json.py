# -*- coding: utf-8 -*-
#
# convert CSV to two JSON files (language list and feature list)
#
import sys, os
import codecs
import json
from argparse import ArgumentParser
from collections import defaultdict

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from csv_utils import UnicodeReader
from json_utils import load_json_stream

def main():
    parser = ArgumentParser()
    parser.add_argument("csv", metavar="WALS_LANG", default=None)
    parser.add_argument("langs", metavar="MERGED", default=None)
    parser.add_argument("flist", metavar="FLIST", default=None)
    args = parser.parse_args()

    legend = ("wals_code", "iso_code", "glottocode", "name", "latitude", "longitude", "genus", "family", "macroarea", "countrycodes")
    csv = UnicodeReader(open(args.csv, 'rb'))
    row = csv.next()
    idx2fid = {}
    fid2struct = []
    for k in legend:
        row.pop(0)
    for i, k in enumerate(row):
        wals_id = k[0 : k.index(" ")]
        idx2fid[i] = len(fid2struct)
        # idx: WALS index
        # id: our own index (will be changed by filtering)
        fid2struct.append({ "idx": i, "fid": i, "wals_id": wals_id, "name": k, "label2vid": {} })

    wals_list = []
    for row in csv:
        lang = { "features": {}, "source": "WALS" }
        for k in legend:
            v = row.pop(0)
            lang[k] = v

        if lang["family"] == "other":
            # "Sign Languages" or "Creoles and Pidgins"
            continue

        for idx, fid in idx2fid.iteritems():
            k = row[idx]
            if len(k) > 0:
                walsval = int(k[0 : k.index(" ")])
                wals_id = fid2struct[fid]["wals_id"]
                lang["features"][wals_id] = walsval - 1
                fid2struct[fid]["label2vid"][k] = walsval - 1
        wals_list.append(lang)

    for fid, struct in enumerate(fid2struct):
        struct["vid2label"] = []
        for l, vid in struct["label2vid"].iteritems():
            # unused feature values for Feature 141A: Writing Systems
            while vid >= len(struct["vid2label"]):
                struct["vid2label"].append(None)
            struct["vid2label"][vid] = l

    with codecs.getwriter("utf-8")(open(args.flist, 'w')) as f:
        f.write("%s\n" % json.dumps(fid2struct))

    with codecs.getwriter("utf-8")(open(args.langs, 'w')) as f:
        for lang in wals_list:
            f.write("%s\n" % json.dumps(lang))

if __name__ == "__main__":
    main()
