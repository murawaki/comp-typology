#!/bin/env python
# -*- coding: utf-8 -*-
#
# Merge Glottolog tree and WALS languages
#
import sys
import os
import codecs
import re
from cPickle import load, dump
from argparse import ArgumentParser

from newick_tree import Node

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from json_utils import load_json_file, load_json_stream


def attach_lang(node, glotto_code2lang):
    # lower nodes take priority over attachment
    for child in node.children:
        attach_lang(child, glotto_code2lang)

    m = attach_lang.code_re.search(node.name)
    if m:
        code = m.group(1)
        if code in glotto_code2lang:
            if len(glotto_code2lang[code]) > 1:
                # create child nodes under the target node and attach languages to them
                sys.stderr.write("creating child nodes because glottocode %s is shared by %d langs\n" % (code, len(glotto_code2lang[code])))
                for i, lang in enumerate(glotto_code2lang[code]):
                    child = Node("%d_%d" % (node._id, i))
                    child.name = node.name
                    child.lang = lang
                    child.is_virtual_child = True
                    node.is_virtual_parent = True
                    child.parent = node
                    node.children.append(child)
                    lang["used"] = True
            else:
                lang = glotto_code2lang[code][0]
                if "used" in lang and lang["used"] == True:
                    pass
                    # sys.stderr.write("glottocode: %s already appeared\n" % (code))
                else:
                    node.lang = lang
                    lang["used"] = True
            # if len(node.children) > 0:
            #     sys.stderr.write("NOTE: attach data to an internal node: %s\n" % node.name)
    else:
        sys.stderr.write("node without glottocode: %s\n" % node.name)

attach_lang.code_re = re.compile(r" \[([a-z0-9]{4}[0-9]{4})\]")

def shrink_tree(node):
    children2 = []
    for child in node.children:
        c = shrink_tree(child)
        if c > 0:
            children2.append(child)
    node.children = children2
    if len(node.children) <= 0:
        if hasattr(node, "lang"):
            # becomes a leaf
            return 1
        else:
            # sys.stderr.write("node %s removed (non-leaf without children)\n" % node.name)
            return 0
    else:
        if hasattr(node, "lang"):
            sys.stderr.write("creating a node for %s because the intermediate node is attached\n" % node.name)
            child = Node("%d_c" % (node._id))
            child.name = node.name
            child.lang = node.lang
            child.is_virtual_child = True
            node.is_virtual_parent = True
            del node.lang
            child.parent = node
            node.children.append(child)
        else:
            if len(node.children) > 1:
                # valid intermediate node
                if len(node.children) > 2:
                    sys.stderr.write("\tnode %s has %d children (%d nodes need to be added to bifurcate)\n" % (node.name, len(node.children), len(node.children) - 2))
            else:
                # skip intermediate node by overriding the node
                # sys.stderr.write("skipping node %s (non-leaf with one child)\n" % node.name)
                parent = None
                if hasattr(node, "parent"):
                    parent = node.parent
                for k, v in node.children[0].__dict__.iteritems():
                    setattr(node, k, v)
                for child in node.children:
                    child.parent = node
                if parent:
                    node.parent = parent
        return 1

def get_feature_coverage(tree):
    countlist = []
    def _get_feature_coverage(node):
        if hasattr(node, "lang"):
            catvect = node.lang["catvect"]
            if len(countlist) <= 0:
                for i in xrange(len(catvect)):
                    countlist.append(0)
            for i, v in enumerate(catvect):
                if v >= 0:
                    countlist[i] += 1
        for child in node.children:
            _get_feature_coverage(child)
    _get_feature_coverage(tree)
    c = len(filter(lambda x: x > 0, countlist))
    return float(c) / len(countlist)


def main():
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = ArgumentParser()
    parser.add_argument("--lthres", dest="lthres", metavar="FLOAT", type=float, default=0.0,
                        help="eliminate trees with higher rate of missing values [0,1]")
    parser.add_argument("langs", metavar="LANG", default=None)
    parser.add_argument("tree", metavar="TREE", default=None)
    parser.add_argument("out", metavar="OUTPUT", default=None)
    args = parser.parse_args()

    trees = load(open(args.tree, 'rb'))
    langs = [lang for lang in load_json_stream(open(args.langs))]
    glotto_code2lang = {}
    for lang in langs:
        if lang["glottocode"]:
            if lang["glottocode"] in glotto_code2lang:
                glotto_code2lang[lang["glottocode"]].append(lang)
            else:
                glotto_code2lang[lang["glottocode"]] = [lang]
        else:
            sys.stderr.write("dropping lang without glottocode: %s\n" % lang["name"])
    for tree in trees:
        attach_lang(tree, glotto_code2lang)
    for code, langlist in glotto_code2lang.iteritems():
        for lang in langlist:
            if "used" in lang and lang["used"] == True:
                del lang["used"]
            else:
                sys.stderr.write("glottocode never appeared in trees: %s\n" % code)
    trees2 = []
    for tree in trees:
        c = shrink_tree(tree)
        if c > 0:
            trees2.append(tree)
    sys.stderr.write("# of trees: %d -> %d\n" % (len(trees), len(trees2)))
    if args.lthres > 0.0:
        trees3 = []
        for tree in trees2:
            c = get_feature_coverage(tree)
            if c >= args.lthres:
                trees3.append(tree)
        sys.stderr.write("# of trees (%f thres): %d -> %d\n" % (args.lthres, len(trees2), len(trees3)))
        trees2 = trees3

    with open(args.out, "w") as f:
        dump(trees2, f)


if __name__ == "__main__":
    main()
