from statsmodels.tools.data_interface import (get_ndim, flatten_list, is_recarray, transpose, drop_recarray_column,
                                              recarray_to_pandas, is_col_vector, to_pandas, NumPyInterface)

import pandas as pd
import numpy as np


def to_categorical(data, col=None, dictnames=False, drop=False):

    ndim = get_ndim(data)
    to_type = np.recarray

    if type(data) == list:
        data = flatten_list(data)

    if is_recarray(data):

        if ndim > 1:
            if col is None:
                raise ValueError('Col must not be None for multi dimensional recarray')
            else:
                if type(col) == int:
                    col = data.dtype.names[col]

                rec_column = data[col]
                to_dummies = transpose(rec_column)

                to_drop = drop_recarray_column(data, col)
                to_drop = recarray_to_pandas(to_drop)

        else:
            to_drop = None

            if data.dtype.names is not None:
                name = data.dtype.names[0]
                to_dummies = data[name]
            else:
                to_dummies = data

            if is_col_vector(to_dummies):
                to_dummies = transpose(to_dummies)

        to_dummies = pd.Series(to_dummies)

    else:
        to_type = type(data)
        to_dummies = to_pandas(data)

        if ndim > 1:
            to_drop = to_dummies.drop(col, 1)

            if isinstance(col, int):
                to_dummies = to_dummies.iloc[:, col]
            elif isinstance(col, str):
                to_dummies = to_dummies.loc[:, col]

        else:
            to_drop = None

    source_data = to_pandas(data)
    dummies = pd.get_dummies(to_dummies)

    interface = NumPyInterface(external_type=to_type)

    if not drop:
        result = pd.concat([source_data, dummies], axis=1)
    else:
        result = pd.concat([to_drop, dummies], axis=1)

    return interface.from_statsmodels(result)