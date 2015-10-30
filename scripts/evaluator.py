# -*- coding: utf-8 -*-

import sys
import codecs
import json
import numpy as np
import random


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x)) # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T # ndim = 2


class BinaryFeatureList(object):
    def __init__(self, lang_struct):
        for k,v in lang_struct.iteritems():
            setattr(self, k, v)
        self.binvect = np.array(self.binvect, dtype=np.int32)


class CategoricalFeatureList(object):
    def __init__(self, mv_catvect, evaluator, has_missing_values=True, fid_freq=None):
        self.evaluator = evaluator
        self.orig = mv_catvect
        self.catvect = np.copy(self.orig)
        self.mv_list = []
        self.has_missing_values = has_missing_values

        if self.has_missing_values:
            # randomly initialize missing values
            for fid, v in enumerate(self.orig):
                if v < 0: # missing value
                    self.mv_list.append(fid)
                    if fid_freq:
                        v = rand_partition(fid_freq[fid])
                    else:
                        size = self.evaluator.map_cat2bin[fid,1]
                        v = np.random.randint(0, size)
                    self.catvect[fid] = v
        self.binvect = self.evaluator.cat2bin(self.catvect)

    def propose_mv_constrained(self, fid=-1):
        catvect2 = np.copy(self.catvect)
        binvect2 = np.copy(self.binvect)
        if fid < 0:
            fid = np.random.random_integers(0, high=len(self.mv_list) - 1) # inclusive
        old_v = catvect2[fid]
        bidx, size = self.evaluator.map_cat2bin[fid]
        new_v = np.random.random_integers(0, high=size - 2)
        if new_v >= old_v:
            new_v += 1
        catvect2[fid] = new_v
        binvect2[bidx + old_v] = 0
        binvect2[bidx + new_v] = 1
        return (binvect2, catvect2)



class BaseEvaluator(object):
    def __init__(self, binsize, dims=50, eta=0.01, _lambda=0.001, penalty=None):
        self.binsize = binsize
        self.dims = dims
        self.weight_list = [("We", "gWe"), ("be", "gbe"), ("Wd", "gWd"), ("bd", "gbd"), ("Ws", "gWs")]

        # init_e = 0.01 / self.binsize
        init_e = 4 * np.sqrt(6.0 / (self.binsize + self.dims)) # per DeepLearning tutorial
        self.We = np.random.uniform(-init_e, init_e, (self.dims, self.binsize))   # encode
        self.be = np.random.uniform(-init_e, init_e, self.dims)
        # self.Wd = np.random.uniform(-init_e, init_e, (self.binsize, self.dims)) # decode
        self.Wd = self.We.T  # tied weight
        self.bd = np.random.uniform(-init_e, init_e, self.binsize)
        self.Ws = np.random.uniform(-init_e, init_e, self.dims) # evaluation

        self.gWe = 1e-45 * np.ones((self.dims, self.binsize))
        self.gbe = 1e-45 * np.ones(self.dims)
        self.gWd = 1e-45 * np.ones((self.binsize, self.dims))
        self.gbd = 1e-45 * np.ones(self.binsize)
        self.gWs = 1e-45 * np.ones(self.dims)

        # SGD-related params
        self.penalty = penalty
        self.scale = 1.0
        self.eta = eta
        self.eta_start = eta
        self._lambda = _lambda # L2 regularization term
        self.time = 1

    def update_weight(self, delta):
        cdiff = 0.0

        # update weights using AdaGrad
        for _type, _gtype in self.weight_list:
            if _type not in delta:
                continue
            diff = self.scale * delta[_type]
            if np.fabs(diff).sum() < 1E-100: # fail-safe
                continue
            W = getattr(self, _type)
            G = getattr(self, _gtype)
            G += diff * diff
            S = np.sqrt(G)
            if self.penalty is None or self._lambda <= 0.0:
                # without regularization
                udiff = self.eta * diff / S
                W += udiff
                cdiff += np.fabs(udiff).sum()
            elif self.penalty == 'l2':
                # L2 regularization
                denom = self.eta * self._lambda + S
                uterm = self.eta * (diff / denom)
                W *= (S / denom)
                W += uterm
                # ignore the shrinkage term
                cdiff += np.fabs(uterm).sum()
                # sys.stderr.write("%f\n" % (np.fabs(self.eta * diff / np.sqrt(G)).sum() - np.fabs(uterm).sum()))
            else:
                # L1 regularization
                g = diff / S
                a = W + self.eta * g
                W2 = np.sign(a) * np.maximum(np.fabs(a) - self._lambda * g, 0.0)
                setattr(self, _type, W2)
                cdiff += self.eta * np.fabs(g).sum() # approximate

        self.time += 1
        # if self.time % 100 == 0:
        # #    self.scale /= 1.0 + self._lambda # self._lambda * self.eta
        # #     self.eta /= 1.0 + (10.0 / self.time)
        #     if self.eta < self.eta_start * 0.0001:
        #         self.eta = self.eta_start * 0.0001
        return cdiff

    def encode(self, binvect):
        # sigmoid
        return sigmoid(self.scale * (np.dot(self.We, binvect) + self.be))
        # # tanh
        # return np.tanh(self.scale * (np.dot(self.We, binvect) + self.be))
        # # linear
        # return self.scale * (np.dot(self.We, binvect) + self.be)

    def decode(self, hvect):
        # sigmoid
        return sigmoid(self.scale * (np.dot(self.Wd, hvect) + self.bd))
        # # tanh
        # return np.tanh(self.scale * (np.dot(self.Wd, hvect) + self.bd))
        # # linear
        # return self.scale * (np.dot(self.Wd, hvect) + self.bd)

    def calc_score(self, hvect):
        return self.scale * np.dot(self.Ws, hvect)

    def calc_delta_autoencoder(self, binvect, delta=None, count=1.0):
        if delta is None:
            delta = {
                "We": np.zeros((self.dims, self.binsize)),
                "be": np.zeros(self.dims),
                }
        if "Wd" not in delta:
            delta["Wd"] = np.zeros((self.binsize, self.dims))
            delta["bd"] = np.zeros(self.binsize)

        hvect = self.encode(binvect)
        binvect2 = self.decode(hvect)

        # TODO: softmax (1-of-K constraints)-based error

        # cross-entropy (sigmoid)
        error2 = binvect - binvect2

        delta["Wd"] += count * np.outer(error2, hvect)
        delta["bd"] += count * error2

        error3 = self.scale * np.dot(self.Wd.T, error2)
        error4 = error3 * (hvect * (1.0 - hvect)) # sigmoid
        # error4 = error3 * (1.0 - hvect * hvect) # tanh
        # error4 = error3 # linear
        delta["We"] += count * np.outer(error4, binvect)
        delta["be"] += count * error4
        return (delta, np.dot(error2, error2) / 2)

    def calc_delta_scorer(self, binvect, hvect, delta=None, count=1.0):
        if delta is None:
            delta = {
                "We": np.zeros((self.dims, self.binsize)),
                "be": np.zeros(self.dims),
                }
        if "Ws" not in delta:
            delta["Ws"] = np.zeros(self.dims)

        delta["Ws"] += count * hvect

        error = self.scale * self.Ws * (hvect * (1.0 - hvect)) # sigmoid
        # error = self.scale * self.Ws * (1.0 - hvect * hvect) # tanh
        # error = self.scale * self.Ws # linear
        delta["We"] += count * np.outer(error, binvect)
        delta["be"] += count * error
        return delta

    def _calc_partition_function(self, burn_in=100, interval=10, samples=100, initial=None):
        # # we ommit the N term because it causes underflow
        # if self.logZ is not None and np.random.uniform(0.0, 1.0) > 0.01:
        #     return self.logZ

        if initial is None:
            current = np.random.random_integers(0, 1, size=self.binsize)
        else:
            current = initial
        current_hvect = self.encode(current)
        current_score = self.calc_score(current_hvect)

        score_vect = np.empty(samples)
        for _iter in xrange(burn_in):
            current, current_hvect, current_score, is_accepted = self._sample_unconstrained(current, current_hvect, current_score)
        score_vect[0] = current_score
        for _iter1 in xrange(samples - 1):
            for _iter2 in xrange(0, interval):
                current, current_hvect, current_score, is_accepted = self._sample_unconstrained(current, current_hvect, current_score)
            score_vect[_iter1 + 1] = current_score
        m = np.max(score_vect)
        self.logZ = np.log(np.exp(score_vect - m).sum()) + m - np.log(samples)
        return self.logZ

    def _sample_unconstrained(self, current, current_hvect, current_score):
        proposed = self._propose_unconstrained(current)
        proposed_hvect = self.encode(proposed)
        proposed_score = self.calc_score(proposed_hvect)
        e = np.exp([current_score, proposed_score] - np.max([current_score, proposed_score]))
        if e[0] < np.random.uniform(0.0, e.sum()):
            # accepted
            return (proposed, proposed_hvect, proposed_score, True)
        else:
            return (current, current_hvect, current_score, False)

    def _propose_unconstrained(self, binvect, do_copy=True):
        if do_copy:
            binvect = np.copy(binvect)
        fid = np.random.random_integers(0, high=self.binsize-1)
        binvect[fid] = 1 if binvect[fid] == 0 else 0
        return binvect


class NestedEvaluator(object):
    def init_nested(self, binsize, dims=50, dims2=10):
        # super(NestedEvaluator, self).__init__(binsize, dims=dims, eta=eta, _lambda=_lambda, penalty=penalty)
        self.dims2 = dims2
        self.weight_list.append(("Wh", "gWh"))
        self.weight_list.append(("bh", "gbh"))
    
        init_e = 4 * np.sqrt(6.0 / (self.binsize + self.dims)) # per DeepLearning tutorial
        self.Ws = np.random.uniform(-init_e, init_e, self.dims2)
        self.Wh = np.random.uniform(-init_e, init_e, (self.dims2, self.dims))
        self.bh = np.random.uniform(-init_e, init_e, self.dims2)

        self.gWs = 1e-45 * np.ones(self.dims2)
        self.gWh = 1e-45 * np.ones((self.dims2, self.dims))
        self.gbh = 1e-45 * np.ones(self.dims2)

    def calc_score(self, hvect):
        return self.scale * np.dot(self.Ws, sigmoid(self.scale * (np.dot(self.Wh, hvect) + self.bh)))

    def calc_delta_scorer(self, binvect, hvect, delta=None, count=1.0):
        if delta is None:
            delta = {
                "We": np.zeros((self.dims, self.binsize)),
                "be": np.zeros(self.dims),
                }
        if "Ws" not in delta:
            delta["Ws"] = np.zeros(self.dims2)
            delta["Wh"] = np.zeros((self.dims2, self.dims))
            delta["bh"] = np.zeros(self.dims2)

        hvect2 = sigmoid(self.scale * (np.dot(self.Wh, hvect) + self.bh))
        delta["Ws"] += count * hvect2

        # error = self.scale * self.Ws * (1.0 - hvect2 * hvect2) # tanh
        error = self.scale * self.Ws * (hvect2 * (1.0 - hvect2)) # sigmoid
        # error = self.scale * self.Ws # linear
        delta["Wh"] += count * np.outer(error, hvect)
        delta["bh"] += count * error

        error2 = self.scale * np.dot(self.Wh.T, error) * (hvect * (1.0 - hvect))
        delta["We"] += count * np.outer(error2, binvect)
        delta["be"] += count * error2
        return delta


class BinaryFeatureListEvaluator(BaseEvaluator):
    def __init__(self, binsize, dims=50, eta=0.01, _lambda=0.001, penalty=None, is_empty=False):
        if is_empty:
            return

        super(BinaryFeatureListEvaluator, self).__init__(binsize=binsize, dims=dims, eta=eta, _lambda=_lambda, penalty=penalty)
        self.logZ = None

    def set_freqvect(self, freqvect):
        # for frequency-based initialization
        self.freqvect = freqvect

    def _denumpy(self):
        obj = {
            "_class": type(self).__name__,
            "dims": self.dims,
            "binsize": self.binsize,
            "weight_list": self.weight_list,
            }
        if hasattr(self, "freqvect"):
            obj["freqvect"] = self.freqvect.tolist()
        for _type, _gtype in self.weight_list:
            obj[_type] = (self.scale * getattr(self, _type)).tolist()
        return obj

    def dumps(self):
        return json.dumps(self._denumpy())

    @classmethod
    def loads(self, dat):
        struct = json.loads(dat)
        return globals()[struct["_class"]]._numpy(struct)

    @classmethod
    def _numpy(self, struct):
        if "_class" in struct:
            obj = globals()[struct["_class"]](None, is_empty=True)
        else:
            # backward-compatibility
            obj = BinaryFeatureListEvaluator(None, is_empty=True)
        obj.dims = struct["dims"]
        obj.binsize = struct["binsize"]
        obj.scale = 1.0
        obj.weight_list = struct["weight_list"]
        for _type, _gtype in obj.weight_list:
            setattr(obj, _type, np.array(struct[_type]))
        if "freqvect" in struct:
            obj.freqvect = np.array(struct["freqvect"])
        return obj

    def binarize(self, binvect2):
        # normalize
        binvect = np.zeros(self.binsize, dtype=np.int32)
        for i,v in enumerate(binvect2):
            if v >= 0.5:
                binvect[i] = 1
            else:
                binvect[i] = 0
        return binvect

    def init_rand_binvect(self):
        if hasattr(self, "freqvect"):
            return np.greater(self.freqvect, np.random.rand(self.binsize)) - 0
        else:
            r = np.random.random_integers(1, 500) # control the frequency of 0-valued elements
            return np.random.random_integers(0, r, size=self.binsize) / r

    def train_scorer(self, tnode, burn_in=100, interval=10, psamples=100, nsamples=100, Cscore=1.0, delta=None):
        # 2 types of negative samples
        #
        # 1. samples from around positive samples
        # 2. samples from a long-lasting MCMC chain

        # 1. samples from around positive samples
        current = tnode.binvect
        current_hvect = self.encode(current)
        current_score = self.calc_score(current_hvect)
        # calc the expected count
        for _iter1 in xrange(nsamples):
            for _iter2 in xrange(0, interval):
                current, current_hvect, current_score, is_accepted = self._sample_unconstrained(current, current_hvect, current_score)
            delta = self.calc_delta_scorer(current, current_hvect, delta=delta, count=-Cscore / (2 * nsamples))

        # 2. samples from a long-lasting MCMC chain
        if not hasattr(tnode, "rand_binvect") or np.random.uniform(0.0, 1.0) < 0.00005:
            sys.stderr.write("reset rand binvect\n")
            tnode.rand_binvect = self.init_rand_binvect()
        rand_hvect = self.encode(tnode.rand_binvect)
        rand_score = self.calc_score(rand_hvect)
        for _iter1 in xrange(nsamples):
            for _iter2 in xrange(0, interval):
                tnode.rand_binvect, rand_hvect, rand_score, is_accepted = self._sample_unconstrained(tnode.rand_binvect, rand_hvect, rand_score)
            delta = self.calc_delta_scorer(tnode.rand_binvect, rand_hvect, delta=delta, count=-Cscore / (2 * nsamples))

        # calc the expected count
        current_hvect = self.encode(tnode.binvect)
        current_score = self.calc_score(current_hvect)
        delta = self.calc_delta_scorer(tnode.binvect, current_hvect, delta=delta, count=Cscore)
        return delta


class NestedBinaryFeatureListEvaluator(NestedEvaluator, BinaryFeatureListEvaluator):
    def __init__(self, binsize, dims=50, dims2=10, eta=0.01, _lambda=0.001, penalty=None, is_empty=False):
        if is_empty:
            return

        super(NestedBinaryFeatureListEvaluator, self).__init__(binsize, dims=dims, eta=eta, _lambda=_lambda, penalty=penalty, is_empty=is_empty)
        self.init_nested(binsize, dims=dims, dims2=dims2)
        self.logZ = None

    def _denumpy(self):
        obj = BinaryFeatureListEvaluator._denumpy(self)
        obj["dims2"] = self.dims2
        return obj

    @classmethod
    def _numpy(self, struct):
        obj = BinaryFeatureListEvaluator._numpy(self, struct)
        obj.dims2 = struct["dims2"]
        return obj


class CategoricalFeatureListEvaluator(BaseEvaluator):
    def __init__(self, fid2struct, dims=50, eta=0.01, _lambda=0.001, penalty=None, is_empty=False):
        if is_empty:
            return

        self.fid2struct = fid2struct
        self.catsize = len(fid2struct)
        binsize = 0
        self.map_cat2bin = np.empty((self.catsize, 2), dtype=np.int32) # (first elem. idx, size)
        for fid, fnode in enumerate(fid2struct):
            size = len(fnode["vid2label"])
            self.map_cat2bin[fid] = [binsize, size]
            binsize += size

        BaseEvaluator.__init__(self, binsize, dims=dims, eta=eta, _lambda=_lambda, penalty=penalty)
        self.map_bin2cat = np.empty((self.binsize, 2), dtype=np.int32) # (fid, idx)

        idx = 0
        for fid, fnode in enumerate(fid2struct):
            for v, flabel in enumerate(fnode["vid2label"]):
                self.map_bin2cat[idx] = [fid, v]
                idx += 1

        self.logZ = None

    def _denumpy(self):
        obj = {
            "_class": type(self).__name__,
            "dims": self.dims,
            "catsize": self.catsize,
            "binsize": self.binsize,
            "map_cat2bin": self.map_cat2bin.tolist(),
            "map_bin2cat": self.map_bin2cat.tolist(),
            "fid2struct": self.fid2struct,
            "weight_list": self.weight_list,
            }
        for _type, _gtype in self.weight_list:
            obj[_type] = (self.scale * getattr(self, _type)).tolist()
        return obj

    def dumps(self):
        return json.dumps(self._denumpy())

    @classmethod
    def loads(self, dat):
        struct = json.loads(dat)
        return self._numpy(struct)

    @classmethod
    def _numpy(self, struct):
        if "_class" in struct:
            obj = globals()[struct["_class"]](None, is_empty=True)
        else:
            # backward-compatibility
            obj = CategoricalFeatureListEvaluator(None, is_empty=True)
        obj.dims = struct["dims"]
        obj.catsize = struct["catsize"]
        obj.binsize = struct["binsize"]
        obj.map_cat2bin = np.array(struct["map_cat2bin"], dtype=np.int32)
        obj.map_bin2cat = np.array(struct["map_bin2cat"], dtype=np.int32)
        obj.fid2struct = struct["fid2struct"]
        obj.scale = 1.0
        obj.weight_list = struct["weight_list"]
        for _type, _gtype in obj.weight_list:
            setattr(obj, _type, np.array(struct[_type]))
        return obj

    def binarize(self, binvect2):
        # normalize
        binvect = np.zeros(self.binsize, dtype=np.int32)
        for fid in xrange(self.catsize):
            boffset, bsize = self.map_cat2bin[fid]
            bidx = boffset + np.argmax(binvect2[boffset:boffset+bsize])
            binvect[bidx] = 1
        return binvect

    def train_scorer(self, tnode, burn_in=100, interval=10, psamples=100, nsamples=100, Cscore=1.0, delta=None):
        ## 4 types of negative samples
        #
        ## 1. samples from around positive samples without categorical constraints
        # 2. samples from around positive samples with categorical constraints
        ## 3. samples from a long-lasting MCMC chain without categorical constraints
        # 4. samples from a long-lasting MCMC chain with categorical constraints

        # # # binvect may violate categorical constraints
        # current = tnode.binvect
        # current_hvect = self.encode(current)
        # current_score = self.calc_score(current_hvect)
        # # calc the expected count
        # for _iter1 in xrange(nsamples):
        #     for _iter2 in xrange(0, interval):
        #         current, current_hvect, current_score, is_accepted = self._sample_unconstrained(current, current_hvect, current_score)
        #     delta = self.calc_delta_scorer(current, current_hvect, delta=delta, count=-Cscore / (4 * nsamples))

        # # binvect never violates categorical constraints
        current_catvect = tnode.catvect
        current_binvect = tnode.binvect
        current_hvect = self.encode(current_binvect)
        current_score = self.calc_score(current_hvect)
        # calc the expected count
        for _iter1 in xrange(nsamples):
            for _iter2 in xrange(0, interval):
                current_score, current_catvect, current_binvect, current_hvect, is_accepted = \
                    self._sample_all_constrained(current_catvect, current_binvect, current_hvect, current_score)
            delta = self.calc_delta_scorer(current_binvect, current_hvect, delta=delta, count=-Cscore / (4 * nsamples))

        # # random MCMC without categorical constraints
        # if not hasattr(tnode, "rand_binvect") or np.random.uniform(0.0, 1.0) < 0.00005:
        #     sys.stderr.write("reset rand binvect\n")
        #     r = np.random.random_integers(1, 20)  # control the frequency of 0-valued elements
        #     tnode.rand_binvect = np.random.random_integers(0, r, size=self.binsize) / r
        # rand_hvect = self.encode(tnode.rand_binvect)
        # rand_score = self.calc_score(rand_hvect)
        # for _iter1 in xrange(nsamples):
        #     for _iter2 in xrange(0, interval):
        #         tnode.rand_binvect, rand_hvect, rand_score, is_accepted = self._sample_unconstrained(tnode.rand_binvect, rand_hvect, rand_score)
        #     delta = self.calc_delta_scorer(tnode.rand_binvect, rand_hvect, delta=delta, count=-Cscore / (4 * nsamples))

        # random MCMC with categorical constraints
        if not hasattr(tnode, "randnode") or np.random.uniform(0.0, 1.0) < 0.0001:
            sys.stderr.write("reset randnode\n")
            tnode.randnode = CategoricalFeatureList(-1 * np.ones(self.catsize, dtype=np.int32), self) 
        rand_binvect = tnode.randnode.binvect
        rand_hvect = self.encode(rand_binvect)
        rand_score = self.calc_score(rand_hvect)
        for _iter1 in xrange(nsamples):
            for _iter2 in xrange(0, interval):
                rand_score, tnode.randnode.catvect, tnode.randnode.binvect, rand_hvect, is_accepted = \
                    self._sample_all_constrained(tnode.randnode.catvect, tnode.randnode.binvect, rand_hvect, rand_score)
            delta = self.calc_delta_scorer(tnode.randnode.binvect, rand_hvect, delta=delta, count=-Cscore / (4 * nsamples))

        # calc the expected count
        current_hvect = self.encode(tnode.binvect)
        current_score = self.calc_score(current_hvect)
        if tnode.has_missing_values:
            for _iter1 in xrange(psamples - 1):
                for _iter2 in xrange(0, interval):
                    current_score, curent_hvect, is_accepted = self._sample_mv_constrained(tnode, current_hvect, current_score)
                delta = self.calc_delta_scorer(tnode.binvect, current_hvect, delta=delta, count=Cscore / psamples)
        else:
            delta = self.calc_delta_scorer(tnode.binvect, current_hvect, delta=delta, count=Cscore)
        return delta


    def _sample_all_constrained(self, current_catvect, current_binvect, current_hvect, current_score):
        proposed_catvect, proposed_binvect = self._propose_all_constrained(current_catvect, current_binvect)
        proposed_hvect = self.encode(proposed_binvect)
        proposed_score = self.calc_score(proposed_hvect)
        e = np.exp([current_score, proposed_score] - np.max([current_score, proposed_score]))
        if e[0] < np.random.uniform(0.0, e.sum()):
            # accepted
            return (proposed_score, proposed_catvect, proposed_binvect, proposed_hvect, True)
        else:
            return (current_score, current_catvect, current_binvect, current_hvect, False)

    def _propose_all_constrained(self, catvect, binvect):
        catvect2 = np.copy(catvect)
        binvect2 = np.copy(binvect)
        fid = np.random.random_integers(0, high=self.catsize - 1) # exclusive
        old_v = catvect2[fid]
        bidx, size = self.map_cat2bin[fid]
        new_v = np.random.random_integers(0, high=size - 2)
        if new_v >= old_v:
            new_v += 1
        catvect2[fid] = new_v
        binvect2[bidx + old_v] = 0
        binvect2[bidx + new_v] = 1
        return (catvect2, binvect2)

    def _sample_mv_constrained(self, tnode, current_hvect, current_score, fid=-1):
        binvect2, catvect2 = tnode.propose_mv_constrained(fid=fid)
        proposed_hvect = self.encode(binvect2)
        proposed_score = self.calc_score(proposed_hvect)
        e = np.exp([current_score, proposed_score] - np.max([current_score, proposed_score]))
        if e[0] < np.random.uniform(0.0, e.sum()):
            # accepted
            tnode.binvect = binvect2
            tnode.catvect = catvect2
            return (proposed_score, proposed_hvect, True)
        else:
            return (current_score, current_hvect, False)

    def cat2bin(self, catvect):
        # # -1 or 1, not 0 or 1
        # binvect = -1 * np.ones(self.binsize, dtype=np.int32)
        binvect = np.zeros(self.binsize, dtype=np.int32)
        for fid, v in enumerate(catvect):
            if v < 0:
                raise Exception("negative value in category vector: %d" % v)
            if v >= self.map_cat2bin[fid][1]:
                raise Exception("out-of-range error in category vector: %d" % v)
            binvect[self.map_cat2bin[fid][0] + v] = 1
        return binvect

    def bin2cat(self, binvect):
        catvect = -1 * np.ones(self.catsize, dtype=np.int32)
        for idx, v in enumerate(binvect):
            if v:
                s = self.map_bin2cat[idx]
                catvect[s[0]] = s[1]
        return catvect


class NestedCategoricalFeatureListEvaluator(NestedEvaluator, CategoricalFeatureListEvaluator):
    def __init__(self, fid2struct, dims=50, dims2=10, eta=0.01, _lambda=0.001, penalty=None, is_empty=False):
        if is_empty:
            return

        self.fid2struct = fid2struct
        self.catsize = len(fid2struct)
        binsize = 0
        self.map_cat2bin = np.empty((self.catsize, 2), dtype=np.int32) # (first elem. idx, size)
        for fid, fnode in enumerate(fid2struct):
            size = len(fnode["vid2label"])
            self.map_cat2bin[fid] = [binsize, size]
            binsize += size

        super(NestedCategoricalFeatureListEvaluator, self).__init__(fid2struct, dims=dims, eta=eta, _lambda=_lambda, penalty=penalty, is_empty=is_empty)
        self.init_nested(binsize, dims=dims, dims2=dims2)
        self.map_bin2cat = np.empty((self.binsize, 2), dtype=np.int32) # (fid, idx)

        idx = 0
        for fid, fnode in enumerate(fid2struct):
            for v, flabel in enumerate(fnode["vid2label"]):
                self.map_bin2cat[idx] = [fid, v]
                idx += 1

        self.logZ = None

    def _denumpy(self):
        obj = CategoricalFeatureListEvaluator._denumpy(self)
        obj["dims2"] = self.dims2
        return obj

    @classmethod
    def _numpy(self, struct):
        obj = CategoricalFeatureListEvaluator._numpy(self, struct)
        obj.dims2 = struct["dims2"]
        return obj
