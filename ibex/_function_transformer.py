from __future__ import absolute_import

from six import string_types
import pandas as pd
from sklearn import base

from ._base import FrameMixin
from ._utils import verify_x_type, verify_y_type


__all__ = []


def _process_cols(cols):
    if cols is None:
        return None

    return [cols] if isinstance(cols, string_types) else list(cols)


# Tmp Ami - add kw_args, inverse shit
class FunctionTransformer(base.BaseEstimator, base.TransformerMixin, FrameMixin):
    """
    Transforms using a function.

    Arguments:

        func: One of:

            * ``None``
            * a callable
            * a step

        in_cols: One of:

            * ``None``
            * a string
            * a list of strings

        out_cols:

        pass_y: Boolean indicating whether to pass the ``y`` argument to

        kw_args:

    Returns:

        An :py:class:`sklearn.preprocessing.FunctionTransformer` object.
    """
    def __init__(self, func=None, in_cols=None, out_cols=None, pass_y=None, kw_args=None):
        FrameMixin.__init__(self)

        params = {
            'func': func,
            'in_cols': in_cols,
            'out_cols': out_cols,
            'pass_y': pass_y,
            'kw_args': kw_args,
        }

        self.set_params(**params)

    def fit(self, X, y=None):
        """
        Fits the transformer using ``X`` (and possibly ``y``).

        Returns:

            ``self``
        """
        verify_x_type(X)
        verify_y_type(y)

        self.x_columns = X.columns

        if self.in_cols is not None:
            Xt = X[self.in_cols]
        else:
            Xt = X

        if self.func is None:
            return self

        if isinstance(self.func, FrameMixin):
            if self.pass_y:
                self.func.fit(Xt, y)
            else:
                self.func.fit(Xt)

        return self

    def fit_transform(self, X, y=None):
        """
        Fits the transformer using ``X`` (and possibly ``y``), and transforms, in one
        step if possible

        Returns:

            Transformed data.
        """
        verify_x_type(X)
        verify_y_type(y)

        if not isinstance(self.func, FrameMixin) or not hasattr(self.func, 'fit_transform'):
            if self.pass_y:
                return self.fit(X, y).transform(X, y)
            return self.fit(X).transform(X)

        self.x_columns = X.columns

        Xt = X[self.x_columns]

        in_cols = _process_cols(self.in_cols)

        if in_cols is not None:
            Xt = Xt[in_cols]

        if self.func is None:
            res = Xt
        elif isinstance(self.func, FrameMixin):
            if self.pass_y:
                res = self.func.fit_transform(Xt, y)
            else:
                res = self.func.fit_transform(Xt)
        else:
            res = pd.DataFrame(self.func(Xt, **self.kw_args), index=Xt.index)

        return self.__process_res(Xt, res)

    def transform(self, X, y=None):
        """
        Returns:

            Transformed data.
        """
        verify_x_type(X)
        verify_y_type(y)

        Xt = X[self.x_columns]

        in_cols = _process_cols(self.in_cols)

        if in_cols is not None:
            Xt = Xt[in_cols]

        if self.func is None:
            res = Xt
        elif isinstance(self.func, FrameMixin):
            if self.pass_y:
                res = self.func.transform(Xt, y)
            else:
                res = self.func.transform(Xt)
        else:
            res = pd.DataFrame(self.func(Xt, **self.kw_args), index=Xt.index)

        return self.__process_res(Xt, res)

    def __process_res(self, Xt, res):
        in_cols = _process_cols(self.in_cols)
        out_cols = _process_cols(self.out_cols)
        if out_cols is not None:
            res_cols = out_cols
        elif in_cols is not None:
            res_cols = in_cols
        else:
            res_cols = Xt.columns

        res = res.copy()
        res.columns = res_cols
        return res



