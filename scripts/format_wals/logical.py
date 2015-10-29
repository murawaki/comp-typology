# -*- coding: utf-8 -*-
#
# fill logically determined feature values
#
import sys, os
import codecs
import json
from argparse import ArgumentParser
from collections import defaultdict

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from json_utils import load_json_file, load_json_stream

def get_struct_by_wid(fid2struct, wid):
    for i, struct in enumerate(fid2struct):
        if struct["wals_id"] == wid:
            return struct
    return None

def main():
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = ArgumentParser()
    parser.add_argument("langs_in", metavar="LANGS_IN", default=None)
    parser.add_argument("flist_in", metavar="FLIST_IN", default=None)
    parser.add_argument("langs_out", metavar="LANGS_OUT", default=None)
    parser.add_argument("flist_out", metavar="FLIST_OUT", default=None)
    args = parser.parse_args()

    fid2struct = load_json_file(args.flist_in)
    langs =[lang for lang in load_json_stream(open(args.langs_in))]

    specs = [
        { "flist": ["144D", "144E", "144F", "144G"],
          "target": ["2 SVO", "7 No dominant order"], },
    # 144D The Position of Negative Morphemes in SVO Languages
    # 144E Multiple Negative Constructions in SVO Languages
    # 144F Obligatory Double Negation in SVO languages
    # 144G Optional Double Negation in SVO languages
    #   only applicable to SVO languages
        { "flist": ["144L", "144M", "144N", "144O"],
          "target": ["1 SOV", "7 No dominant order"], },
    # 144L The Position of Negative Morphemes in SOV Languages
    # 144M Multiple Negative Constructions in SOV Languages
    # 144N Obligatory Double Negation in SOV languages
    # 144O Optional Double Negation in SOV languages
    #   only applicable to SOV languages
        { "flist": ["144T", "144U", "144V", "144W", "144X"],
          "target": ["3 VSO", "4 VOS", "7 No dominant order"], },
    # 144T: The Position of Negative Morphemes in Verb-Initial Languages
    # 144U Double negation in verb-initial languages
    # 144V Verb-Initial with Preverbal Negative
    # 144W Verb-Initial with Negative that is Immediately Postverbal or between Subject and Object
    # 144X Verb-Initial with Clause-Final Negative
    #   only applicable to VSO and VOS languages
        { "flist": ["144Y"],
          "target": ["5 OVS", "6 OSV", "7 No dominant order"] }
    # 144Y The Position of Negative Morphemes in Object-Initial Languages
    #   only applicable to OVS and OSV languages
    ]

    for spec in specs:
        spec["uvid"] = []
        flist2 = []
        for wid in spec["flist"]:
            struct = get_struct_by_wid(fid2struct, wid)
            if struct is None:
                sys.stderr.write("skip undefined feature: %s\n" % wid)
                continue
            flist2.append(wid)

            struct["label2vid"]["-1 Undefined"] = len(struct["vid2label"])
            struct["vid2label"].append("-1 Undefined")
            spec["uvid"].append(struct["label2vid"]["-1 Undefined"])
        spec["flist"] = flist2

    # Feature 81A: Order of Subject, Object and Verb
    SOV_WID = "81A"
    sov_struct = get_struct_by_wid(fid2struct, SOV_WID)
    filled = 0
    for lang in langs:
        if SOV_WID not in lang["features"]:
            continue
        sov_val = sov_struct["vid2label"][lang["features"][SOV_WID]]
        for spec in specs:
            if sov_val in spec["target"]:
                continue
            for i, wid in enumerate(spec["flist"]):
                if wid in lang["features"]:
                    sys.stderr.write("illogical feature values: %s\t%s\t%s\t%s\n" % (lang["name"], sov_val, wid, lang["features"][wid]))
                else:
                    lang["features"][wid] = spec["uvid"][i]
                    filled += 1
    sys.stderr.write("%d elements fileld\n" % filled)

    with codecs.getwriter("utf-8")(open(args.flist_out, 'w')) as f:
        f.write("%s\n" % json.dumps(fid2struct))

    with codecs.getwriter("utf-8")(open(args.langs_out, 'w')) as f:
        for lang in langs:
            f.write("%s\n" % json.dumps(lang))

if __name__ == "__main__":
    main()
