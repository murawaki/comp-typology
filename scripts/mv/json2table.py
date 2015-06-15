# -*- coding: utf-8 -*-
#
# output TSV file for imput_mca.r
#
import sys, os
import codecs
import json

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from json_utils import load_json_file


def main(fpath, src, dst):
    fid2struct = load_json_file(fpath)
    langlist = load_json_file(src)

    with codecs.getwriter("utf-8")(open(dst, 'w')) as f:
        rv = "\t".join([feature["name"] for feature in fid2struct])
        f.write(rv + "\n")
        for name, flist in langlist.iteritems():
            rv = name + "\t"
            for fid, v in enumerate(flist):
                if v == -1:
                    rv += "NA\t"
                else:
                    rv += str(int(v)) + "\t"
            rv = rv[0:len(rv) - 1]
            f.write(rv + "\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
