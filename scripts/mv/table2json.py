# -*- coding: utf-8 -*-

import sys
import codecs
import json

def main(src, dst):
    obj = {}
    with open(src) as f:
        f.readline() # ignore the header
        while True:
            l = f.readline()
            if not l:
                break
            l = l.rstrip()
            a = l.split("\t")
            label = a.pop(0)
            obj[label] = [int(s) for s in a]
    with codecs.getwriter("utf-8")(open(dst, 'w')) as f:
        f.write(json.dumps(obj))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

