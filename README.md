DocumentFeatureSelection
==========================

# what's this?

This is set of feature selection codes from text data.
(About feature selection, see [here](http://nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html) or [here](http://stackoverflow.com/questions/13603882/feature-selection-and-reduction-for-text-classification))

The Feature selection is really important when you use machine learning metrics on natural language data.
The natural language data usually contains a lot of noise information, thus machine learning metrics are weak if you don't process any feature selection.
(There is some exceptions of algorithms like _Decision Tree_ or _Random forest_ . They have feature selection metric inside the algorithm itself)

The feature selection is also useful when you observe your text data.
With the feature selection, you can get to know which features really contribute to specific labels.

Please visit [project page on github](https://github.com/Kensuke-Mitsuzawa/DocumentFeatureSelection).

If you find any bugs and you report it to github issue, I'm glad.

Any pull-requests are welcomed.

## Supporting methods

This package provides you some feature selection metrics.
Currently, this package supports following feature selection methods

* TF-IDF
* Pointwise mutual information (PMI)
* Strength of Association (SOA)
* Bi-Normal Separation (BNS)

## Contribution of this package

* Easy interface for pre-processing
* Easy interface for accessing feature selection methods
* Fast speed computation thanks to sparse matrix and multi-processing

# Overview of methods

## TF-IDF

This method, in fact, just calls `TfidfTransformer` of the scikit-learn.

See [scikit-learn document](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) about detailed information.

## PMI

PMI is calculated by correlation between _feature_ (i.e. token) and _category_ (i.e. label).
Concretely, it makes _cross-table_ (or called _contingency table_) and calculates joint probability and marginal probability on it.

To know more, see [reference](https://www.eecis.udel.edu/~trnka/CISC889-11S/lectures/philip-pmi.pdf)

In python world, [NLTK](http://www.nltk.org/howto/collocations.html) and [Other package](https://github.com/Bollegala/svdmi) also provide PMI.
Check them and choose based on your preference and usage.


## SOA

SOA is improved feature-selection method from PMI.
PMI is weak when feature has low word frequency.
SOA is based on PMI computing, however, it is feasible on such low frequency features.
Moreover, you can get anti-correlation between features and categories.

In this package, SOA formula is from following paper,

`Saif Mohammad and Svetlana Kiritchenko, "Using Hashtags to Capture Fine Emotion Categories from Tweets", Computational Intelligence, 01/2014; 31(2).`

```
SOA(w, e)\ =\ log_2\frac{freq(w, e) * freq(\neg{e})}{freq(e) * freq(w, \neg{e})}
```

Where

* freq(w, e) is the number of times _w_ occurs in an unit(sentence or document) with label _e_
* freq(w,¬e) is the number of times _w_ occurs in units that does not have the label _e_
* freq(e) is the number of units having the label _e_
* freq(¬e) is the number of units having NOT the label _e_

## BNS

BNS is a feature selection method for binary class data.
There is several methods available for binary class data, such as _information gain (IG)_, _chi-squared
(CHI)_, _odds ratio (Odds)_.
 
The problem is when you execute your feature selection on skewed data.
These methods are weak for such skewed data, however, _BNS_ is feasible only for skewed data.
The following paper shows how BNS is feasible for skewed data.

```Lei Tang and Huan Liu, "Bias Analysis in Text Classification for Highly Skewed Data", 2005```

or 

```George Forman, "An Extensive Empirical Study of Feature Selection Metrics for Text Classification",Journal of Machine Learning Research 3 (2003) 1289-1305```
 

# Requirement

* Python 3.x(checked under Python 3.5)


# Setting up

## install

`python setup.py install`

### Note

You might see error message during running this command, such as

```
We failed to install numpy automatically. Try installing numpy manually or Try anaconda distribution.
```

This is because `setup.py` tries to instal numpy and scipy with `pip`, however it fails. 
We need numpy and scipy before we install `scikit-learn`.

In this case, you take following choice

* You install `numpy` and `scipy` manually
* You use `anaconda` python distribution. Please visit [their site](https://www.continuum.io/downloads).

# Example

```python
input_dict = {
    "label_a": [
        ["I", "aa", "aa", "aa", "aa", "aa"],
        ["bb", "aa", "aa", "aa", "aa", "aa"],
        ["I", "aa", "hero", "some", "ok", "aa"]
    ],
    "label_b": [
        ["bb", "bb", "bb"],
        ["bb", "bb", "bb"],
        ["hero", "ok", "bb"],
        ["hero", "cc", "bb"],
    ],
    "label_c": [
        ["cc", "cc", "cc"],
        ["cc", "cc", "bb"],
        ["xx", "xx", "cc"],
        ["aa", "xx", "cc"],
    ]
}

from DocumentFeatureSelection import interface
interface.run_feature_selection(input_dict, method='pmi', use_cython=True).convert_score_matrix2score_record()
```
Then, you get the result

```python
[{'score': 0.14976146817207336, 'label': 'label_c', 'feature': 'bb', 'frequency': 1.0}, ...]
```

See scripts in `examples/`

# For developers

You could set up dev environment with docker-compose.

This command runs test with the docker container.

```bash
$ cd tests/
$ docker-compose build
$ docker-compose up
```

