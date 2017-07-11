import inspect
import types
import functools

import numpy as np
import pandas as pd
import sklearn
from sklearn import pipeline

import _feature_union
import _frame_mixin


def _to_step(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return _Adapter(result)
    return wrapper


def _xy_wrapper(method, self):
    @functools.wraps(method)
    def xy_wrapped(step, X, *args, **kwargs):
        self._set_x(X)
        param_X = self._x(X)
        if len(args) > 0:
            y = self._y(args[0])
            args = args[1: ]
            ret = method(param_X, y, *args, **kwargs)
        else:
            ret = method(param_X, *args, **kwargs)

        if isinstance(ret, np.ndarray):
            print ret.shape
            if len(ret.shape) == 1:
                return pd.Series(ret, index=X.index)
            if len(ret.shape) == 2:
                print X, X.index, X.columns
                print pd.DataFrame(ret, index=X.index, columns=X.columns)
                return pd.DataFrame(ret, index=X.index, columns=X.columns)

        if ret == step:
            return self

        return ret
    return xy_wrapped


def _x_wrapper(method, self):
    @functools.wraps(method)
    def x_wrapped(step, X, *args, **kwargs):
        print X, args, kwargs
        self._set_x(X)
        ret = method(self._x(X), *args, **kwargs)

        for _ in range(20):
            print 'x', ret

        if isinstance(ret, np.ndarray):
            print ret.shape
            if len(ret.shape) == 1:
                return pd.Series(ret, index=X.index)
            if len(ret.shape) == 2:
                print X, X.index, X.columns
                print pd.DataFrame(ret, index=X.index, columns=X.columns)
                return pd.DataFrame(ret, index=X.index, columns=X.columns)

        if ret == step:
            return self

        return ret
    return x_wrapped


class _Adapter(_frame_mixin.FrameMixin):
    """
    Adapts a step to a pandas based step.

    The resulting step takes pd entities; if needed, it strips
        them and passes them to the adapted step as numpy entities.

    Arguments:
        a step or pipeline.
    """
    def __init__(self, step):
        class BaseAdded(_Adapter, type(step)):
            pass
        self.__class__ = BaseAdded
        self.__name__ = '_Adapter'

        _frame_mixin.FrameMixin.__init__(self)

        self._step = step

        for method_name in dir(step):
            try:
                method = getattr(step, method_name)
            except AttributeError:
                continue

            print 'here', method_name

            # Tmp Ami
            if not callable(method):
                continue

            try:
                args = inspect.getargspec(method).args
            except TypeError:
                continue

            if args[: 3] == ['self', 'X', 'y']:
                print('setting', method_name)
                self.__setattr__(method_name, types.MethodType(_xy_wrapper(method, self), step))
                print self.__dict__
                continue
            elif args[: 2] == ['self', 'X']:
                print('setting', method_name)
                self.__setattr__(method_name, types.MethodType(_x_wrapper(method, self), step))
                print self.__dict__
                continue
            else:
                # Tmp Ami - add here
                continue

    def __getattr__(self, name):
        print('getattr', name)
        result = getattr(self._step, name)
        if callable(result):
            result = _to_step(result)
        return result

    def _x(self, x):
        return x if _frame_mixin.FrameMixin.is_subclass(self._step) else x.as_matrix()

    def _y(self, y):
        if y is None:
            return None
        return y if _frame_mixin.FrameMixin.is_subclass(self._step) else y.values

    # Tmp Ami - check if next two are needed
    def get_params(self, deep=True):
        """
        See sklearn.base.BaseEstimator.get_params
        """
        return self._step.get_params(deep)

    def set_params(self, *params):
        """
        See sklearn.base.BaseEstimator.set_params
        """
        return self._step.set_params(*params)


def frame(step):
    return _Adapter(step)