# -*- coding: utf-8 -*-
#
# unsupervised training of typology evaluator (categorical)
#
import sys
import codecs
import json
import numpy as np
import random
from argparse import ArgumentParser

from json_utils import load_json_file
from evaluator import CategoricalFeatureList, CategoricalFeatureListEvaluator, NestedCategoricalFeatureListEvaluator

def cum_error(binvect_list, evaluator):
    rv = 0.0
    for binvect in binvect_list:
        binvect2 = evaluator.decode(evaluator.encode(binvect))
        binvect3 = evaluator.binarize(binvect2)
        rv += np.absolute(binvect - binvect3).sum()
    return rv

def train(tnode_list, evaluator, _iter=100, minibatch=10, _iter_offset=1, Cscore=0.1, interval=5, psamples=5, nsamples=5):
    for _i in xrange(_iter):
        random.shuffle(tnode_list)

        cdiff = 0.0
        count = 0
        error = 0.0
        delta = None
        for tnode in tnode_list:
            delta = evaluator.train_scorer(tnode, burn_in=0, interval=interval, psamples=psamples, nsamples=nsamples, delta=delta, Cscore=Cscore)
            delta, error_each = evaluator.calc_delta_autoencoder(tnode.binvect, delta=delta)
            error += error_each

            count += 1
            if count % minibatch == 0:
                cdiff += evaluator.update_weight(delta)
                delta = None
        if count % minibatch != 0:
            cdiff += evaluator.update_weight(delta)
            delta = None
        sys.stderr.write("AE\titer %d: cdiff: %f\tt_error: %f\tc_error: %f\n" % \
                             (_i + _iter_offset, cdiff, error / len(tnode_list),
                              cum_error([tnode.binvect for tnode in tnode_list], evaluator) / float(len(tnode_list))))
        shuffle_randnode(tnode_list)
        
def shuffle_randnode(node_list):
    i = len(node_list) - 1
    while i > 0:
        r = np.random.random_integers(0, high=i)
        node_list[i].randnode, node_list[r].randnode = node_list[r].randnode, node_list[i].randnode
        i -= 1
    return node_list

def save(evaluator, modelfile, tnode_list):
    with codecs.getwriter("utf-8")(open(modelfile, 'w')) as f:
        f.write(evaluator.dumps())

def main():
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr)

    parser = ArgumentParser()
    parser.add_argument("--nested", dest="nested", action="store_true", default=False)
    parser.add_argument("-s", "--seed", dest="seed", metavar="INT", type=int, default=None,
                        help="random seed")
    parser.add_argument("-d", "--dims", dest="dims", metavar="INT", type=int, default=200,
                        help="number of dimensions")
    parser.add_argument("--dims2", dest="dims2", metavar="INT", type=int, default=10,
                        help="number of dimensions")
    parser.add_argument("-i", "--iter", dest="_iter", metavar="INT", type=int, default=20000,
                        help="number of dimensions")
    parser.add_argument("--eta", dest="eta", metavar="FLOAT", type=float, default=0.01,
                        help="SGD parameter")
    parser.add_argument("--penalty", dest="penalty", metavar="l1 or l2", default=None,
                        help="regularization l1 or l2 (default None)")
    parser.add_argument("--lambda", dest="_lambda", metavar="FLOAT", type=float, default=0.0,
                        help="L2 regularization term")
    parser.add_argument("--Cscore", dest="Cscore", metavar="FLOAT", type=float, default=0.1,
                        help="balance between autoencoder and scorer")
    parser.add_argument("--minibatch", dest="minibatch", metavar="INT", type=int, default=10,
                        help="minibatch size (default: 10)")
    parser.add_argument("--interval", dest="interval", metavar="INT", type=int, default=10)
    parser.add_argument("--psamples", dest="psamples", metavar="INT", type=int, default=10)
    parser.add_argument("--nsamples", dest="nsamples", metavar="INT", type=int, default=10)
    parser.add_argument("langs", metavar="LANG", default=None)
    parser.add_argument("fid2struct", metavar="FID2STRUCT", default=None)
    parser.add_argument("model", metavar="MODEL", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    langlist_orig = load_json_file(args.langs)
    fid2struct = load_json_file(args.fid2struct)
    modelfile = args.model

    if args.nested:
        evaluator = NestedCategoricalFeatureListEvaluator(fid2struct, dims=args.dims, dims2=args.dims2, eta=args.eta, _lambda=args._lambda, penalty=args.penalty)
    else:
        evaluator = CategoricalFeatureListEvaluator(fid2struct, dims=args.dims, eta=args.eta, _lambda=args._lambda, penalty=args.penalty)

    # mv_count = 0
    train_total = 0
    langlist = []
    for label, lang in langlist_orig.iteritems():
        tnode = CategoricalFeatureList(lang, evaluator, has_missing_values=False)
        tnode.label = label
        train_total += 1
        langlist.append(tnode)

    sys.stderr.write("# of catvect elemts: %d\n" % evaluator.catsize)
    # sys.stderr.write("missing value rate: %f\n" %  (mv_count / (float(train_total * evaluator.catsize))))
    sys.stderr.write("# of binvect elems: %d\n" % evaluator.binsize)
    sys.stderr.write("# of training instances: %d\n" % train_total)
    sys.stderr.write("Cscore: %d\n" % args.Cscore)
    sys.stderr.write("# of hidden dims: %d\n" % evaluator.dims)
    if args.nested:
        sys.stderr.write("# of hidden dims2: %d\n" % evaluator.dims2)
    sys.stderr.write("interval, psamples, nsamples: (%d, %d, %d)\n" % (args.interval, args.psamples, args.nsamples))
    sys.stderr.write("SGD/Adagrad eta: %f\n" % evaluator.eta)
    sys.stderr.write("penalty: %s\n" % evaluator.penalty)
    sys.stderr.write("lambda: %f\n" % evaluator._lambda)
                     
    _iter_remaining = args._iter
    _iter_count=0
    while _iter_remaining > 0:
        _iter_each = min(1000, _iter_remaining)
        train(langlist, evaluator, _iter=_iter_each, _iter_offset=_iter_count, minibatch=args.minibatch, Cscore=args.Cscore,
              interval=10, psamples=10, nsamples=10)
        _iter_remaining -= 1000
        _iter_count += 1000
        save(evaluator, modelfile, langlist)

if __name__ == "__main__":
    main()
