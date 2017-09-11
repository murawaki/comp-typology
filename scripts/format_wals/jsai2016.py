# -*- coding: utf-8 -*-
#
# select languages and convert them to a NEXUS matrix
# used in a JSAI paper
#
import sys, os
import codecs
import json
import numpy as np
import random
from argparse import ArgumentParser
from collections import Counter

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from json_utils import load_json_file, load_json_stream

def main():
    parser = ArgumentParser()
    parser.add_argument("langs", metavar="LANG", default=None)
    parser.add_argument("flist", metavar="FLIST", default=None)
    args = parser.parse_args()

    fthres = 5
    target_langs = {
        "Japanese": True,
        "Korean": True,
        "Evenki": True,
        "Khalkha": True,
        "Turkish": True,
        "Ainu": True,
        "Nivkh": True,
        "Chukchi": True,
        "Yukaghir (Kolyma)": True,
        "Ket": True,
        "Mandarin": True,
        "Burmese": True,
        "Tagalog": True,
        "Khmer": True,
    }
    flist = load_json_file(args.flist)
    wals_id2feature = {}
    for feature in flist:
        wals_id2feature[feature["wals_id"]] = feature
    
    langstructs = []
    fcounts = Counter()
    for lang in load_json_stream(open(args.langs)):
        if lang["name"] not in target_langs:
            continue
        langstructs.append(lang)
        for k in lang["features"].iterkeys():
            fcounts[k] += 1
    for k, v in fcounts.items():
        if v < fthres:
            del fcounts[k]
    used_features = fcounts.keys()
    used_features.sort()

    sys.stderr.write("%d features: %s\n" % (len(used_features), " ".join(used_features)))

    for lang in langstructs:
        binrep = []
        for wals_id in used_features:
            knum = len(wals_id2feature[wals_id]["vid2label"])
            if wals_id in lang["features"]:
                binrep2 = ["0"] * knum
                binrep2[lang["features"][wals_id]] = "1"
            else:
                binrep2 = ["?"] * knum
            binrep += binrep2
        lang["bin"] = binrep
    nchar = len(langstructs[0]["bin"])

    # nexus
    rv = "#nexus\r\nBEGIN DATA;\r\nDIMENSIONS ntax=%d nchar=%d;\r\nFORMAT\r\n\tdatatype=standard\r\n\tsymbols=\"01\"\r\n\tmissing=?\r\n\tgap=-\r\n\tinterleave=NO\r\n;\r\nMATRIX\n\n" % (len(langstructs), nchar)
    for lang in langstructs:
        name_normalized = lang["name"].replace(" ", "_").replace("(", "").replace(")", "")
        rv += ("%s\t%s\r" % (name_normalized, "".join(lang["bin"])))
    rv += ";\r\nEND;"
    sys.stdout.write(rv)

if __name__ == "__main__":
    main()
