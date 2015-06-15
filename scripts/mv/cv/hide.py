# -*- coding: utf-8 -*-
#
import sys, os
import codecs
import json
import random

sys.path.insert(1, os.path.join(os.path.join(sys.path[0], os.path.pardir), os.path.pardir))
from json_utils import load_json_file


def main(src, dst, cvmap_file, cvn):
    langlist = load_json_file(src)
    cvmap = load_json_file(cvmap_file)

    for label, fid in cvmap[cvn]:
        langlist[label][fid] = -1

    with codecs.getwriter("utf-8")(open(dst, 'w')) as f:
        f.write(json.dumps(langlist))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
