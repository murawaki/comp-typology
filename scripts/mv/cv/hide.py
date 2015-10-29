# -*- coding: utf-8 -*-
#
import sys, os
import codecs
import json
import random

sys.path.insert(1, os.path.join(os.path.join(sys.path[0], os.path.pardir), os.path.pardir))
from json_utils import load_json_file, load_json_stream


def main(src, dst, cvmap_file, cvn):
    langs = [lang for lang in load_json_stream(open(src))]
    cvmap = load_json_file(cvmap_file)

    wals_code2lang = {}
    for lang in langs:
        wals_code2lang[lang["wals_code"]] = lang

    for wals_code, wals_id in cvmap[cvn]:
        del wals_code2lang[wals_code]["features"][wals_id]

    with codecs.getwriter("utf-8")(open(dst, 'w')) as f:
        for lang in langs:
            f.write("%s\n" % json.dumps(lang))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
