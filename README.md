document-feature-selection
==========================

# what's this?

This is set of feature extraction(a.k.a feature selection) codes from text data.
(About feature selection, see [here](http://nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html) or [here](http://stackoverflow.com/questions/13603882/feature-selection-and-reduction-for-text-classification))

The Feature selection is really important when you use machine learning metrics on natural language data.
The natural language data usually contains a lot of noise information, thus machine learning metrics are weak if you don't process any feature selection.
(There is some exceptions of algorithms like _Decision Tree_ or _Random forest_ . They have feature selection metric inside the algorithm itself)

The feature selection is also useful when you observe your text data.
With the feature selection, you can get to know which features really contribute to specific labels.


## Supporting methods

This package provides you some feature selection metrics.
Currently, this package supports following feature selection methods

* TF-IDF
* Probabilistic mutual information(PMI)

## Contribution of this package

* Easy interface for pre-processing
* Easy interface for accessing feature selection methods
* Fast speed computation thanks to sparse matrix and multi-processing


# Requirement

* Python 3.x(checked under Python 3.5)


# Setting up

## install

`python setup.py install`


# Examples

See scripts in `examples/`


# Change log

## 0.6 2016/04/02

supports PMI and TF-IDF under Python3.x
