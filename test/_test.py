from __future__ import absolute_import


import unittest
import os
from glob import glob
import doctest
import json
import pickle
import inspect
import tempfile

_level = os.getenv('IBEX_TEST_LEVEL')
_level = 1 if _level is None else int(_level)


import six
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import sklearn
from sklearn import preprocessing
from ibex import frame
from ibex.sklearn import _sklearn_ver
try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    from sklearn.utils.validation import NotFittedError # Older Versions
from ibex.sklearn import preprocessing as pd_preprocessing
from sklearn import pipeline
from ibex.sklearn import pipeline as pd_pipeline
from sklearn import base
from ibex.sklearn import decomposition as pd_decomposition
from sklearn import linear_model
from ibex.sklearn import linear_model as pd_linear_model
from sklearn import ensemble
from ibex.sklearn import ensemble as pd_ensemble
from sklearn import feature_selection
from ibex.sklearn import feature_selection as pd_feature_selection
from sklearn import svm
from ibex.sklearn import svm as pd_svm
from sklearn import gaussian_process
from ibex.sklearn import gaussian_process as pd_gaussian_process
from sklearn import feature_selection
from sklearn import neighbors
from ibex.sklearn import neighbors as pd_neighbors
from sklearn import cluster
from ibex.sklearn import cluster as pd_cluster
from sklearn import decomposition
from ibex.sklearn import decomposition as pd_decomposition
try:
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_predict
    from ibex.sklearn.model_selection import GridSearchCV as PdGridSearchCV
    from ibex.sklearn.model_selection import cross_val_predict as pd_cross_val_predict
except (ImportError, NameError):
    from sklearn.cross_validation import cross_val_score
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import cross_val_predict
    from ibex.sklearn.grid_search import GridSearchCV as PdGridSearchCV
    from ibex.sklearn.cross_validation import cross_val_predict as pd_cross_val_predict
from sklearn import datasets
# Tmp Ami - xgboost?
if _level > 0:
    import tensorflow
    from ibex.tensorflow.contrib.keras.wrappers.scikit_learn import KerasClassifier as PdKerasClassifier
    from ibex.tensorflow.contrib.keras.wrappers.scikit_learn import KerasRegressor as PdKerasRegressor
from ibex import *


_tmp_dir = tempfile.mkdtemp()


def _build_regressor_nn():
    model = tensorflow.contrib.keras.models.Sequential()
    model.add(
        tensorflow.contrib.keras.layers.Dense(20, input_dim=4, activation='relu'))
    model.add(
        tensorflow.contrib.keras.layers.Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adagrad')
    return model


def _build_classifier_nn():
    model = tensorflow.contrib.keras.models.Sequential()
    model.add(tensorflow.contrib.keras.layers.Dense(8, input_dim=4, activation='relu'))
    model.add(tensorflow.contrib.keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model


if _level > 0:
    class KerasClassifier(tensorflow.contrib.keras.wrappers.scikit_learn.KerasClassifier, base.ClassifierMixin):
        pass


    class KerasRegressor(tensorflow.contrib.keras.wrappers.scikit_learn.KerasRegressor, base.RegressorMixin):
        pass


_this_dir = os.path.dirname(__file__)


def _load_iris():
    iris = datasets.load_iris()
    features = iris['feature_names']
    iris = pd.DataFrame(
        np.c_[iris['data'], iris['target']],
        columns=features+['class'])
    return iris, features


def _load_digits():
    digits = datasets.load_digits()
    features = ['f%d' % i for i in range(digits['data'].shape[1])]
    digits = pd.DataFrame(
        np.c_[digits['data'], digits['target']],
        columns=features+['digit'])
    digits = digits.sample(frac=0.1).reset_index()
    return digits, features


_dataset_names, _Xs, _ys = [], [], []
_iris, _features = _load_iris()
_dataset_names.append('iris')
_Xs.append(_iris[_features])
_ys.append(_iris['class'])
_Xs.append(_iris[_features])
_ys.append(_iris['class'] == _iris['class'].values[0])
_iris = _iris.copy()
_iris.index = ['i%d' % i for i in range(len(_iris))]
_dataset_names.append('iris_str_index')
_Xs.append(_iris[_features])
_ys.append(_iris['class'])


_estimators = []
_estimators.append((
    preprocessing.StandardScaler(),
    pd_preprocessing.StandardScaler(),
    True))
_estimators.append((
    decomposition.PCA(),
    pd_decomposition.PCA(),
    True))
_estimators.append((
    linear_model.LinearRegression(),
    frame(pd_linear_model.LinearRegression()),
    True))
_estimators.append((
    linear_model.LinearRegression(),
    pd_linear_model.LinearRegression(),
    True))
_estimators.append((
    pipeline.make_pipeline(decomposition.PCA(), linear_model.LinearRegression()),
    pd_decomposition.PCA() | pd_linear_model.LinearRegression(),
    True))
_estimators.append((
    pipeline.make_pipeline(feature_selection.SelectKBest(k=2), decomposition.PCA(), linear_model.LinearRegression()),
    pd_feature_selection.SelectKBest(k=2) | pd_decomposition.PCA() | pd_linear_model.LinearRegression(),
    True))
_estimators.append((
    pipeline.make_pipeline(feature_selection.SelectKBest(k=2), decomposition.PCA(), linear_model.LinearRegression()),
    pd_feature_selection.SelectKBest(k=2) | (pd_decomposition.PCA() | pd_linear_model.LinearRegression()),
    True))
_estimators.append((
    pipeline.make_pipeline(decomposition.PCA(), linear_model.LinearRegression()),
    pd_pipeline.make_pipeline(pd_decomposition.PCA(), pd_linear_model.LinearRegression()),
    True))
_estimators.append((
    linear_model.LogisticRegression(),
    pd_linear_model.LogisticRegression(),
    True))
_estimators.append((
    cluster.KMeans(random_state=42),
    pd_cluster.KMeans(random_state=42),
    True))
_estimators.append((
    cluster.KMeans(random_state=42),
    pickle.loads(pickle.dumps(pd_cluster.KMeans(random_state=42))),
    True))
_estimators.append((
    neighbors.KNeighborsClassifier(),
    pd_neighbors.KNeighborsClassifier(),
    True))
_estimators.append((
    ensemble.GradientBoostingClassifier(random_state=42),
    pd_ensemble.GradientBoostingClassifier(random_state=42),
    True))
_estimators.append((
    pipeline.make_union(decomposition.PCA(n_components=2), feature_selection.SelectKBest(k=1)),
    pd_decomposition.PCA(n_components=2) + pd_feature_selection.SelectKBest(k=1),
    True))
_estimators.append((
    pipeline.FeatureUnion(
        [('pca', decomposition.PCA(n_components=2)), ('kbest', feature_selection.SelectKBest(k=1))],
        transformer_weights={'pca': 3, 'kbest': 4}),
    pd_pipeline.FeatureUnion(
        [('pca', pd_decomposition.PCA(n_components=2)), ('kbest', pd_feature_selection.SelectKBest(k=1))],
        transformer_weights={'pca': 3, 'kbest': 4}),
        True))
_estimators.append((
    pipeline.make_union(decomposition.PCA(n_components=1), decomposition.PCA(n_components=2), feature_selection.SelectKBest(k=1)),
    pd_decomposition.PCA(n_components=1) + pd_decomposition.PCA(n_components=2) + pd_feature_selection.SelectKBest(k=1),
    True))
_estimators.append((
    pipeline.make_union(decomposition.PCA(n_components=1), decomposition.PCA(n_components=2), feature_selection.SelectKBest(k=1)),
    pd_decomposition.PCA(n_components=1) + (pd_decomposition.PCA(n_components=2) + pd_feature_selection.SelectKBest(k=1)),
    True))
_estimators.append((
    pipeline.make_union(decomposition.PCA(n_components=2), feature_selection.SelectKBest(k=1)),
    pd_pipeline.make_union(pd_decomposition.PCA(n_components=2), pd_feature_selection.SelectKBest(k=1)),
    True))
# Tmp Ami - fails without probability=True
_estimators.append((
    pipeline.make_pipeline(
        feature_selection.SelectKBest(k=1), svm.SVC(kernel="linear", random_state=42, probability=True)),
    pd_feature_selection.SelectKBest(k=1) | pd_svm.SVC(kernel="linear", random_state=42, probability=True),
    True))
param_grid = dict(
    C=[0.1, 1, 10])
_estimators.append((
    GridSearchCV(
        svm.SVC(kernel="linear", random_state=42, probability=True),
        param_grid=param_grid,
        verbose=0),
    PdGridSearchCV(
        pd_svm.SVC(kernel="linear", random_state=42, probability=True),
        param_grid=param_grid,
        verbose=0),
    True))
try:
    _estimators.append((
        gaussian_process.GaussianProcessRegressor(),
        pd_gaussian_process.GaussianProcessRegressor(),
        True))
except:
    if _sklearn_ver > 17:
        raise
param_grid = dict(
    logisticregression__C=[0.1, 1, 10])
try:
    _estimators.append((
        GridSearchCV(
            pipeline.make_pipeline(
                linear_model.LogisticRegression()),
            param_grid=param_grid,
            return_train_score=False,
            verbose=0),
        PdGridSearchCV(
            pd_pipeline.make_pipeline(
                pd_linear_model.LogisticRegression()),
            param_grid=param_grid,
            return_train_score=False,
            verbose=0),
        True))
    _estimators.append((
        decomposition.NMF(random_state=42),
        pd_decomposition.NMF(random_state=42),
        True))
except:
    if _sklearn_ver > 17:
        raise


_feature_selectors = []
_feature_selectors.append((
    feature_selection.SelectKBest(k=1),
    pd_feature_selection.SelectKBest(k=1),
    True))
_feature_selectors.append((
    feature_selection.SelectKBest(k=1),
    pickle.loads(pickle.dumps(pd_feature_selection.SelectKBest(k=1))),
    True))
_feature_selectors.append((
    feature_selection.SelectKBest(k=2),
    pd_feature_selection.SelectKBest(k=2),
    True))
_feature_selectors.append((
    feature_selection.SelectPercentile(),
    pd_feature_selection.SelectPercentile(),
    True))
_feature_selectors.append((
    feature_selection.SelectFdr(),
    pd_feature_selection.SelectFdr(),
    True))
_feature_selectors.append((
    feature_selection.SelectFwe(),
    pd_feature_selection.SelectFwe(),
    True))
# Tmp Ami
if False:
    _feature_selectors.append((
        feature_selection.RFE(linear_model.LogisticRegression()),
        pd_feature_selection.RFE(pd_linear_model.LogisticRegression()),
        True))


_keras_estimators = []
if _level > 0:
    _keras_estimators.append((
        KerasClassifier(_build_classifier_nn, verbose=0),
        PdKerasClassifier(_build_classifier_nn, _load_iris()[0]['class'].unique(), verbose=0),
        False))
    _keras_estimators.append((
        KerasRegressor(_build_regressor_nn, verbose=0),
        PdKerasRegressor(_build_regressor_nn, verbose=0),
        False))


class _EstimatorTest(unittest.TestCase):
    pass


def _generate_bases_test(est, pd_est):
    def test(self):
        self.assertTrue(isinstance(pd_est, FrameMixin), pd_est)
        self.assertFalse(isinstance(est, FrameMixin))
        self.assertTrue(isinstance(pd_est, base.BaseEstimator))
        try:
            mixins = [
                base.ClassifierMixin,
                base.ClusterMixin,
                base.BiclusterMixin,
                base.TransformerMixin,
                base.DensityMixin,
                base.MetaEstimatorMixin,
                base.ClassifierMixin,
                base.RegressorMixin]
        except:
            if _sklearn_ver > 17:
                raise
            mixins = [
                base.ClassifierMixin,
                base.ClusterMixin,
                base.BiclusterMixin,
                base.TransformerMixin,
                base.MetaEstimatorMixin,
                base.ClassifierMixin,
                base.RegressorMixin]
        for mixin in mixins:
            self.assertEqual(
                isinstance(pd_est, mixin),
                isinstance(est, mixin),
                mixin)

    return test


def _generate_fit_test(X, y, est, pd_est):
    def test(self):
        pd_est.fit(X, y)
        est.fit(X.as_matrix(), y.values)
    return test


def _generate_feature_importances_test(X, y, est, pd_est):
    def test(self):
        pd_est.fit(X, y)
        est.fit(X.as_matrix(), y.values)
        try:
            importances = est.feature_importances_
        except AttributeError:
            return
        self.assertTrue(isinstance(pd_est.feature_importances_, pd.Series), (est, pd_est))
        np.testing.assert_allclose(pd_est.feature_importances_, importances)
    return test


def _generate_components_test(X, y, est, pd_est):
    def test(self):
        pd_est.fit(X, y)
        est.fit(X.as_matrix(), y.values)
        try:
            comps = est.components_
        except AttributeError:
            return
        self.assertTrue(isinstance(pd_est.components_, pd.DataFrame))
        np.testing.assert_allclose(pd_est.components_, comps)
    return test


def _generate_coef_intercept_test(X, y, est, pd_est):
    def test(self):
        pd_est.fit(X, y)
        est.fit(X.as_matrix(), y.values)
        try:
            coef = est.coef_
            intercept = est.intercept_
        except AttributeError:
            return
        np.testing.assert_allclose(pd_est.coef_, coef)
        np.testing.assert_allclose(pd_est.intercept_, intercept)
    return test


def _generate_fit_matrix_test(X, y, est, pd_est):
    def test(self):
        Y = pd.concat([X.copy(), X.copy()], axis=1)
        Y.columns = [('cc%d' % i) for i in range(len(Y.columns))]
        try:
            est.fit(X.as_matrix(), Y.as_matrix()).predict(X.as_matrix())
        except (ValueError, AttributeError):
            return
        pd_est.fit(X, Y).predict(X)
        try:
            coef = est.coef_
        except AttributeError:
            return
    return test


def _generate_str_repr_test(pd_est):
    def test(self):
        # Tmp Ami - check it starts with Adapter
        self.assertEqual(str(pd_est), repr(pd_est))
    return test


def _generate_array_bad_fit_test(X, y, pd_est):
    def test(self):
        try:
            with self.assertRaises(TypeError):
                pd_est.fit(X.as_matrix(), y)
            with self.assertRaises(TypeError):
                pd_est.fit(X, y.values)
        except:
            print(pd_est)
            raise
    return test


def _generate_index_mismatch_bad_fit_test(X, y, pd_est):
    def test(self):
        global y
        y = y.copy()
        y.index = reversed(X.index.values)
        with self.assertRaises(ValueError):
            pd_est.fit(X, y)
    return test


def _generate_predict_test(X, y, est, pd_est, must_match):
    def test(self):
        self.assertEqual(
            hasattr(est, 'predict'),
            hasattr(pd_est, 'predict'))
        if not hasattr(est, 'predict'):
            return
        pd_y_hat = pd_est.fit(X, y).predict(X)
        self.assertTrue(isinstance(pd_y_hat, pd.Series))
        self.assertTrue(pd_y_hat.index.equals(X.index))
        est.fit(X.as_matrix(), y.values)
        y_hat = est.predict(X.as_matrix())
        if must_match:
            np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_score_test(X, y, est, pd_est, must_match):
    def test(self):
        self.assertEqual(
            hasattr(est, 'score'),
            hasattr(pd_est, 'score'))
        if not hasattr(est, 'score'):
            return
        pd_score = pd_est.fit(X, y).score(X, y)
        # Tmp Ami - what is the type of score?
        est.fit(X.as_matrix(), y.values)
        score = est.score(X.as_matrix(), y.values)
        if must_match:
            np.testing.assert_allclose(pd_score, score)
    return test


def _generate_cross_val_predict_test(X, y, est, pd_est, must_match):
    def test(self):
        self.assertEqual(
            hasattr(est, 'predict'),
            hasattr(pd_est, 'predict'))
        if not hasattr(est, 'predict'):
            return
        pd_y_hat = pd_cross_val_predict(pd_est, X, y)
        self.assertTrue(isinstance(pd_y_hat, pd.Series))
        self.assertTrue(pd_y_hat.index.equals(X.index))
        if must_match:
            y_hat = cross_val_predict(est, X.as_matrix(), y.values)
            np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_score_weight_test(X, y, est, pd_est, must_match):
    def test(self):
        self.assertEqual(
            hasattr(est, 'score'),
            hasattr(pd_est, 'score'))
        if not hasattr(est, 'score'):
            return
        weight = np.abs(np.random.randn(len(y)))
        try:
            pd_score = pd_est.fit(X, y).score(X, y, sample_weight=weight)
        except TypeError:
            pd_score = None
        try:
            est.fit(X.as_matrix(), y.values)
            score = est.score(X.as_matrix(), y.values, sample_weight=weight)
        except TypeError:
            score = None
        if must_match and score is not None:
            self.assertNotEqual(pd_score, None, pd_est)
            self.assertTrue(np.isclose(score, pd_score))
    return test


def _generate_sample_y_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'sample_y'),
            hasattr(pd_est, 'sample_y'))
        if not hasattr(est, 'sample_y'):
            return
        pd_sample = pd_est.fit(X, y).sample_y(X)
        sample = est.fit(X.as_matrix(), y.values).sample_y(X.as_matrix())
        np.testing.assert_allclose(pd_sample, sample)
    return test


def _generate_predict_proba_test(X, y, est, pd_est, must_match):
    def test(self):
        self.assertEqual(
            hasattr(est, 'predict_proba'),
            hasattr(pd_est, 'predict_proba'),
            (est, pd_est))
        if not hasattr(est, 'predict_proba'):
            return
        pd_y_hat = pd_est.fit(X, y).predict_proba(X)
        self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
        self.assertTrue(pd_y_hat.index.equals(X.index))
        # Tmp Ami
        if False and hasattr(est, 'classes_'):
            # Tmp Ami - unreached
            self.assertTrue(pd_y_hat.columns.equals(pd_est.classes_), pd_est)
        est.fit(X.as_matrix(), y.values)
        y_hat = est.predict_proba(X.as_matrix())
        if must_match:
            np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_staged_predict_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'staged_predict'),
            hasattr(pd_est, 'staged_predict'),
            (est, pd_est))
        if not hasattr(est, 'staged_predict'):
            return
        pd_y_hats = pd_est.fit(X, y).staged_predict(X)
        y_hats = est.fit(X.as_matrix(), y.values).staged_predict(X.as_matrix())
        for pd_y_hat, y_hat in zip(pd_y_hats, y_hats):
            self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
            # self.assertTrue(pd_y_hat.columns.equals(pd_est.classes_))
            np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_staged_predict_proba_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'staged_predict_proba'),
            hasattr(pd_est, 'staged_predict_proba'),
            (est, pd_est))
        if not hasattr(est, 'staged_predict_proba'):
            return
        pd_y_hats = pd_est.fit(X, y).staged_predict_proba(X)
        y_hats = est.fit(X.as_matrix(), y.values).staged_predict_proba(X.as_matrix())
        for pd_y_hat, y_hat in zip(pd_y_hats, y_hats):
            self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
            # self.assertTrue(pd_y_hat.columns.equals(pd_est.classes_))
            np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_predict_log_proba_test(X, y, est, pd_est, must_match):
    def test(self):
        self.assertEqual(
            hasattr(est, 'predict_log_proba'),
            hasattr(pd_est, 'predict_log_proba'))
        if not hasattr(est, 'predict_log_proba'):
            return
        pd_y_hat = pd_est.fit(X, y).predict_log_proba(X)
        self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
        self.assertTrue(pd_y_hat.index.equals(X.index))
        # Tmp Ami
        if False and hasattr(est, 'classes_'):
            self.assertTrue(pd_y_hat.columns.equals(pd_est.classes_), pd_est)
        y_hat = est.fit(X.as_matrix(), y.values).predict_log_proba(X.as_matrix())
        np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_fit_predict_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'fit_predict'),
            hasattr(pd_est, 'fit_predict'))
        if not hasattr(est, 'fit_predict'):
            return
        pd_Xt = pd_est.fit_predict(X, y)
        Xt = est.fit_predict(X.as_matrix(), y.values)
        np.testing.assert_allclose(pd_Xt, Xt)
    return test


def _generate_attr_test(X, y, est, pd_est):
    def test(self):
        est.fit(X.as_matrix(), y.values)
        pd_est.fit(X, y)
        self.assertEqual(
            hasattr(est, 'coef_'),
            hasattr(pd_est, 'coef_'))
        if not hasattr(est, 'coef_'):
            return
        np.testing.assert_allclose(est.coef_, pd_est.coef_)
    return test


def _generate_transform_test(X, y, est, pd_est):
    def test(self):
        if not hasattr(est, 'transform'):
            return
        pd_y_hat = pd_est.fit(X, y).transform(X)
        y_hat = est.fit(X.as_matrix(), y.values).transform(X.as_matrix())
        np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


def _generate_inverse_transform_test(X, y, est, pd_est):
    def test(self):
        if not hasattr(est, 'inverse_transform'):
            return
        pd_est.fit(X, y)
        pd_re_X = pd_est.inverse_transform(pd_est.transform(X))
        est.fit(X.as_matrix(), y.values)
        re_X = est.inverse_transform(est.transform(X.as_matrix()))
        np.testing.assert_allclose(pd_re_X, re_X)
    return test


def _generate_fit_transform_test(X, y, est, pd_est):
    def test(self):
        self.assertEqual(
            hasattr(est, 'fit_transform'),
            hasattr(pd_est, 'fit_transform'))
        # sklearn pipeline has a different mechanism for
        # pipeline.Pipeline.fit_transform
        try:
            y_hat = est.fit_transform(X.as_matrix(), y)
        except:
            y_hat = None
        try:
            pd_y_hat = pd_est.fit_transform(X, y)
        except AttributeError:
            pd_y_hat = None
        if pd_y_hat is None:
            self.assertIsNone(y_hat)
        else:
            np.testing.assert_allclose(pd_y_hat, y_hat)
    return test


test_i = 0


for _, pd_est, _ in _estimators + _feature_selectors + _keras_estimators:
    name = type(pd_est).__name__.lower()
    setattr(
        _EstimatorTest,
        'test_str_repr_%s_%d' % (name, test_i),
        _generate_str_repr_test(pd_est))


for est, pd_est, must_match in _estimators + _feature_selectors + _keras_estimators:
    name = type(est).__name__.lower()
    setattr(
        _EstimatorTest,
        'test_bases_%s_%d' % (name, test_i),
        _generate_bases_test(est, pd_est))
    for dataset in zip(_dataset_names, _Xs, _ys):
        dataset_name, X, y = dataset
        name = dataset_name + '_' + type(est).__name__.lower()
        setattr(
            _EstimatorTest,
            'test_fit_%s_%d' % (name, test_i),
            _generate_fit_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_feature_importances_%s_%d' % (name, test_i),
            _generate_feature_importances_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_components_%s_%d' % (name, test_i),
            _generate_components_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_coef_intercept_%s_%d' % (name, test_i),
            _generate_coef_intercept_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_fit_matrix_test_%s_%d' % (name, test_i),
            _generate_fit_matrix_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_array_bad_fit_%s_%d' % (name, test_i),
            _generate_array_bad_fit_test(X, y, pd_est))
        setattr(
            _EstimatorTest,
            'test_index_mismatch_bad_fit_%s_%d' % (name, test_i),
            _generate_index_mismatch_bad_fit_test(X, y, pd_est))
        setattr(
            _EstimatorTest,
            'test_attr_%s_%d' % (name, test_i),
            _generate_attr_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_fit_predict_%s_%d' % (name, test_i),
            _generate_fit_predict_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_predict_%s_%d' % (name, test_i),
            _generate_predict_test(X, y, est, pd_est, must_match))
        setattr(
            _EstimatorTest,
            'test_cross_val_predict_%s_%d' % (name, test_i),
            _generate_cross_val_predict_test(X, y, est, pd_est, must_match))
        setattr(
            _EstimatorTest,
            'test_score_%s_%d' % (name, test_i),
            _generate_score_test(X, y, est, pd_est, must_match))
        setattr(
            _EstimatorTest,
            'test_score_weight_%s_%d' % (name, test_i),
            _generate_score_weight_test(X, y, est, pd_est, must_match))
        setattr(
            _EstimatorTest,
            'test_sample_y__%s_%d' % (name, test_i),
            _generate_sample_y_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_staged_predict_proba_%s_%d' % (name, test_i),
            _generate_staged_predict_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_predict_proba_%s_%d' % (name, test_i),
            _generate_predict_proba_test(X, y, est, pd_est, must_match))
        setattr(
            _EstimatorTest,
            'test_staged_predict_proba_%s_%d' % (name, test_i),
            _generate_staged_predict_proba_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_predict_log_proba_%s_%d' % (name, test_i),
            _generate_predict_log_proba_test(X, y, est, pd_est, must_match))
        setattr(
            _EstimatorTest,
            'test_transform_%s_%d' % (name, test_i),
            _generate_transform_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_inverse_transform_%s_%d' % (name, test_i),
            _generate_inverse_transform_test(X, y, est, pd_est))
        setattr(
            _EstimatorTest,
            'test_fit_transform_%s_%d' % (name, test_i),
            _generate_fit_transform_test(X, y, est, pd_est))

        test_i += 1


def _generate_feature_selection_transform_test(X, y, pd_est):
    def test(self):
        pd_y_hat = pd_est.fit(X, y).transform(X)
        self.assertTrue(isinstance(pd_y_hat, pd.DataFrame))
        for f in pd_y_hat.columns:
            self.assertIn(f, X.columns)
    return test


class _FeatureSelectorTest(unittest.TestCase):
    pass


for est, pd_est, must_match in _feature_selectors:
    for dataset in zip(_dataset_names, _Xs, _ys):
        dataset_name, X, y = dataset
        name = dataset_name + '_' + type(est).__name__.lower()
        setattr(
            _FeatureSelectorTest,
            'test_feature_selection_transform_%s_%d' % (name, test_i),
            _generate_feature_selection_transform_test(X, y, pd_est))

        test_i += 1


class _KerasTest(unittest.TestCase):
    pass


def _generate_keras_history_test(X, y, pd_est):
    def test(self):
        pd_est.fit(X, y).history_

    return test


for est, pd_est, must_match in _keras_estimators:
    for dataset in zip(_dataset_names, _Xs, _ys):
        setattr(
            _KerasTest,
            'test_keras_history_%s_%d' % (name, test_i),
            _generate_keras_history_test(X, y, pd_est))

        test_i += 1


class _FrameTest(unittest.TestCase):
    def test_transform_no_y(self):
        X = pd.DataFrame({'a': [1, 2, 3]})

        pd_Xt = pd_preprocessing.StandardScaler().fit(X).transform(X)
        self.assertTrue(isinstance(pd_Xt, pd.DataFrame))
        Xt = preprocessing.StandardScaler().fit(X).transform(X)
        self.assertFalse(isinstance(Xt, pd.DataFrame))
        np.testing.assert_equal(pd_Xt, Xt)

        pd_Xt = pd_preprocessing.StandardScaler().fit_transform(X)
        self.assertTrue(isinstance(pd_Xt, pd.DataFrame))
        Xt = preprocessing.StandardScaler().fit_transform(X)
        self.assertFalse(isinstance(Xt, pd.DataFrame))
        np.testing.assert_equal(pd_Xt, Xt)

    def test_fit_permute_cols(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = pd_linear_model.LinearRegression().fit(X, y)

        pd_y_hat = pred.predict(X[['b', 'a']])
        self.assertTrue(isinstance(pd_y_hat, pd.Series))
        y_hat = linear_model.LinearRegression().fit(X, y).predict(X)
        self.assertFalse(isinstance(y_hat, pd.Series))
        np.testing.assert_equal(pd_y_hat, y_hat)

    def test_fit_bad_cols(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = pd_linear_model.LinearRegression().fit(X, y)

        y_hat = pred.predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

        X.rename(columns={'a': 'c'}, inplace=True)

        with self.assertRaises(KeyError):
            pred.predict(X)

    def test_object(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        y = pd.Series([1, 2, 3])

        pred = frame(linear_model.LinearRegression()).fit(X, y)

        y_hat = pred.predict(X)
        self.assertTrue(isinstance(y_hat, pd.Series))

        X.rename(columns={'a': 'c'}, inplace=True)

        with self.assertRaises(KeyError):
            pred.predict(X)


class _TransTest(unittest.TestCase):
    def test_trans_none(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        trans().fit(X).transform(X)

        with self.assertRaises(NotFittedError):
            trans().transform(X)

        trans().fit_transform(X)

    def test_trans_none_cols(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        trans(None, 'a').fit(X)

        trans(None, ['a']).fit(X)

        trans(None, 'a').fit(X).transform(X)

        trans(None, ['a']).fit(X).transform(X)

        trans(None, 'a').fit_transform(X)

        trans(None, ['a']).fit_transform(X)

        trans(None, 'a', 'b').fit(X)

        trans(None, 'b', ['a']).fit(X)

        trans(None, 'a', 'b').fit(X).transform(X)

        trans(None, 'b', ['a']).fit(X).transform(X)

        trans(None, 'a', 'b').fit_transform(X)

        trans(None, 'b', ['a']).fit_transform(X)

    def test_trans_none_bad_cols(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})
        bad_X = pd.DataFrame({'a': [1, 2, 3], 'c': [30, 23, 2]})

        with self.assertRaises(KeyError):
            trans(None, 'b').fit(bad_X)

        with self.assertRaises(KeyError):
            trans(None, ['b']).fit(bad_X)

        with self.assertRaises(KeyError):
            trans(None, 'b').fit(X).transform(bad_X)

        with self.assertRaises(KeyError):
            trans(None, ['b']).fit(X).transform(bad_X)

        with self.assertRaises(KeyError):
            trans(None, 'b').fit_transform(bad_X)

        with self.assertRaises(KeyError):
            trans(None, ['b']).fit_transform(bad_X)

    def test_trans_step(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [30, 23, 2]})

        trans(pd_preprocessing.StandardScaler()).fit(X).fit(X)

        trans(pd_preprocessing.StandardScaler()).fit(X).transform(X)

        trans(pd_preprocessing.StandardScaler()).fit(X).fit_transform(X)
        trans(pd_preprocessing.StandardScaler()).fit(X).fit_transform(X, X.a)


class _NBsTest(unittest.TestCase):
    pass


def _generate_nb_tests(name):
    def test(self):
        cmd = 'jupyter-nbconvert --to notebook --execute %s --output %s --ExecutePreprocessor.timeout=7200' % (name, os.path.split(name)[1])

        try:
            self.assertEqual(os.system(cmd), 0)
        except Exception as exc:
            print(cmd, exc)
            # Python2.7 fails on travis, for some reason
            if six.PY3:
                raise
    return test


nb_f_names = list(glob(os.path.join(_this_dir, '../examples/*.ipynb')))
nb_f_names = [n for n in nb_f_names if '.nbconvert.' not in n]
for n in nb_f_names:
    # Tmp Ami - missing boston_cv_preds.ipynb
    with (open(n, encoding='utf-8') if six.PY3 else open(n)) as f:
        cnt = json.loads(f.read(), encoding='utf-8')
    metadata = cnt['metadata']
    if 'ibex_test_level' not in metadata:
        raise KeyError('ibex_test_level missing from metadata of ' + n)
    if _level < int(metadata['ibex_test_level']):
        continue

    test_name = 'test_' + os.path.splitext(os.path.split(n)[1])[0]
    test = _generate_nb_tests(n)
    setattr(_NBsTest, test_name, test)


class _ExternalSubclass(pd_linear_model.LinearRegression):
    pass


class _ExternalComposer(object):
    def __init__(self):
        self.prd = pd_linear_model.LinearRegression()


class _PickleTest(unittest.TestCase):
    def test_direct_single_adapter(self):
        iris, features = _load_iris()

        trn = pd_decomposition.PCA()
        unpickled_trn = pickle.loads(pickle.dumps(trn))

        pca_unpickled = unpickled_trn.fit_transform(iris[features])
        pca = trn.fit_transform(iris[features])
        self.assertTrue(pca_unpickled.equals(pca), (pca.columns, pca_unpickled.columns))

    def test_direct_pipe_adapter(self):
        clf = pd_decomposition.PCA() | pd_linear_model.LinearRegression()
        unpickled_clf = pickle.loads(pickle.dumps(clf))

    def test_grid_search_cv(self):
        from ibex.sklearn.svm import SVC

        iris, features = _load_iris()

        clf = SVC()
        clf = PdGridSearchCV(
            clf,
            {'kernel':('linear', 'rbf'), 'C':[1, 10]},
            n_jobs=1)
        clf.fit(iris[features], iris['class'])

    def test_external_subclass(self):
        e = _ExternalSubclass()
        with self.assertRaises(TypeError):
            pickle.dumps(e)

    def test_external_composition(self):
        e = _ExternalComposer()
        pickle.loads(pickle.dumps(e))

    def test_pickle_fit_adapter(self):
        df_in = pd.DataFrame({'x': [1, np.nan, 3]})
        expected_output = pd.DataFrame({'x': [1., 2., 3.]}, index=df_in.index)

        imp = pd_preprocessing.Imputer(strategy='mean')
        imp.fit(df_in)
        imp_ = pickle.loads(pickle.dumps(imp))
        assert_frame_equal(imp_.transform(df_in), expected_output)


def load_tests(loader, tests, ignore):
    import ibex

    doctest_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

    tests.addTests(doctest.DocTestSuite(__import__('ibex'), optionflags=doctest_flags))
    for mod_name in dir(ibex):
        try:
            mod =__import__('ibex.' + mod_name)
        except ImportError:
            continue
        tests.addTests(doctest.DocTestSuite(mod, optionflags=doctest_flags))

    f_name = os.path.join(_this_dir, '../ibex/sklearn/__init__.py')
    tests.addTests(doctest.DocFileSuite(f_name, module_relative=False, optionflags=doctest_flags))

    f_name = os.path.join(_this_dir, '../ibex/xgboost/__init__.py')
    tests.addTests(doctest.DocFileSuite(f_name, module_relative=False, optionflags=doctest_flags))

    f_name = os.path.join(_this_dir, '../ibex/tensorflow/contrib/keras/wrappers/scikit_learn/__init__.py')
    tests.addTests(doctest.DocFileSuite(f_name, module_relative=False, optionflags=doctest_flags))

    doc_f_names = list(glob(os.path.join(_this_dir, '../docs/source/*.rst')))
    doc_f_names += [os.path.join(_this_dir, '../README.rst')]
    tests.addTests(
        doctest.DocFileSuite(*doc_f_names, module_relative=False, optionflags=doctest_flags))

    doc_f_names = list(glob(os.path.join(_this_dir, '../docs/build/text/*.txt')))
    tests.addTests(
        doctest.DocFileSuite(*doc_f_names, module_relative=False, optionflags=doctest_flags))

    return tests


class _FunctionTransformerTest(unittest.TestCase):
    def test_kwargs(self):
        def f(a, b):
            return a + b

        def g(a, b=2):
            return a + b

        dat = pd.DataFrame({'x': [1, 2, 3, 4, 5]})

        pd_preprocessing.FunctionTransformer(f, kw_args={'b': 2}).fit_transform(dat)
        pd_preprocessing.FunctionTransformer(g).fit_transform(dat)


class _PreprocessingTest(unittest.TestCase):
    def test_polynomial_features(self):
        x = pd.DataFrame({'a': [2], 'b': [3], 'c': [4], 'd': [5]})

        print(pd_preprocessing.PolynomialFeatures(degree=3).fit_transform(x))


if __name__ == '__main__':
    unittest.main()
