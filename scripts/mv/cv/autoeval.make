# -*- mode: Makefile -*-
#
# usage make -f THIS_FILE CV=10
#
LANGS_FILE := ../data/wals/langs.json
FEATURE_FILE := ../data/wals/flist.json
OUTDIR := ../data/cv
LANGS_CVMAP_FILE := $(OUTDIR)/langs.cvmap.json

CV := 10
CV_MAX := $(shell expr $(CV) - 1)
SEED := --seed=2

$(shell mkdir -p $(OUTDIR))


$(LANGS_CVMAP_FILE) : $(LANGS_FILE)
	python mv/cv/make_cvmap.py $(SEED) $(LANGS_FILE) $(FEATURE_FILE) $(LANGS_CVMAP_FILE) $(CV)


# cv_split FILE_PREFIX CV_IDX
define cv_main
$(1).cv$(2).json : $(LANGS_FILE) $(LANGS_CVMAP_FILE)
	python mv/cv/hide.py $(LANGS_FILE) $(1).cv$(2).json $(LANGS_CVMAP_FILE) $(2)

HIDE_LIST += $(1).cv$(2).json

$(1).cv$(2).tsv : $(1).cv$(2).json
	python mv/json2tsv.py $(1).cv$(2).json $(FEATURE_FILE) $(1).cv$(2).tsv

$(1).cv$(2).filled.tsv : $(1).cv$(2).tsv
	R --vanilla -f mv/impute_mca.r --args $(1).cv$(2).tsv $(1).cv$(2).filled.tsv

$(1).cv$(2).filled.json : $(1).cv$(2).filled.tsv
	python mv/tsv2json.py $(1).cv$(2).json $(1).cv$(2).filled.tsv $(FEATURE_FILE) $(1).cv$(2).filled.json

FILLED_LIST += $(1).cv$(2).filled.json
endef


$(foreach i,$(shell seq 0 $(CV_MAX)), \
  $(eval $(call cv_main,$(OUTDIR)/langs,$(i))))


$(OUTDIR)/langs.random.eval : $(LANG_FILE)
	python mv/cv/eval_mv.py $(SEED) --random $(LANGS_FILE) - $(FEATURE_FILE) > $(OUTDIR)/langs.random.eval
EVALS += $(OUTDIR)/langs.random.eval

$(OUTDIR)/langs.freq.eval : $(LANG_FILE) $(HIDE_LIST)
	sh -c 'for i in `seq 0 $(CV_MAX)`; do python mv/cv/eval_mv.py $(SEED) --freq $(LANGS_FILE) $(OUTDIR)/langs.cv$${i}.json $(FEATURE_FILE); done | perl -anle"\$$a+=\$$F[1];\$$b+=\$$F[2]; END{printf \"%f\\t%d\\t%d\\n\", \$$a / \$$b, \$$a, \$$b;}" > $(OUTDIR)/langs.freq.eval'

EVALS += $(OUTDIR)/langs.freq.eval

$(OUTDIR)/langs.mcr.eval : $(LANG_FILE) $(FILLED_LIST)
	sh -c 'for i in `seq 0 $(CV_MAX)`; do python mv/cv/eval_mv.py $(SEED) $(LANGS_FILE) $(OUTDIR)/langs.cv$${i}.filled.json -; done | perl -anle"\$$a+=\$$F[1];\$$b+=\$$F[2]; END{printf \"%f\\t%d\\t%d\\n\", \$$a / \$$b, \$$a, \$$b;}" > $(OUTDIR)/langs.mcr.eval'

EVALS += $(OUTDIR)/langs.mcr.eval


all : $(EVALS)

filled : $(FILLED_LIST)

clean :
	rm -f $(OUTDIR)/lang*
	rmdir --ignore-fail-on-non-empty $(OUTDIR)

.PHONY : all clean filled
