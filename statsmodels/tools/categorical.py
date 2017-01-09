from statsmodels.tools.data_interface import (get_ndim, is_recarray, transpose, is_col_vector, to_pandas,
                                              NumPyInterface)

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


def flatten_list(data):

    assert type(data) == list

    np_data = np.asarray(data)
    shape = np_data.shape

    dims = sum(1 for elem in shape if elem > 1)

    if dims > 2:
        raise ValueError('Data must have no more than two dimensions')
    else:
        slicer = slice_2d(shape)
        data = np_data[slicer].tolist()

    return data


def flatten_array(data):

    assert type(data) in [np.ndarray, np.recarray]

    shape = data.shape

    dims = sum(1 for elem in shape if elem > 1)

    if dims > 2:
        raise ValueError('Data must have no more than two dimensions')
    else:
        slicer = slice_2d(shape)
        data = data[slicer]

    return data


def slice_2d(shape):

    s = []

    for index in shape:
        if index > 1:
            s.append(slice(None))
        else:
            s.append(0)

    return tuple(s)


def drop_recarray_column(rec, name):

    names = list(rec.dtype.names)

    if name in names:
        names.remove(name)

    result = rec[names]

    return result


def recarray_to_pandas(data):

    data_list = []

    if data.dtype.names is None:

        if is_col_vector(data):
            data = transpose(data)

        return pd.Series(data)

    else:

        for name in data.dtype.names:
            col = data[name]

            if is_col_vector(col):
                col = transpose(col)

            data_list.append(pd.Series(col, name=name))

        return pd.concat(data_list, axis=1)
