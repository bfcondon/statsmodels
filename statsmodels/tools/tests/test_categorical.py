from statsmodels.tools.data_interface import to_categorical
from numpy.testing import assert_equal, assert_array_equal

import numpy as np


class TestCategorical(object):

    def __init__(self):

        stringabc = 'abcdefghijklmnopqrstuvwxy'

        self.des = np.random.randn(25, 2)
        self.instr = np.floor(np.arange(10, 60, step=2) / 10)

        x = np.zeros((25, 5))
        x[:5, 0] = 1
        x[5:10, 1] = 1
        x[10:15, 2] = 1
        x[15:20, 3] = 1
        x[20:25, 4] = 1
        self.dummy = x

        structdes = np.zeros((25, 1), dtype=[('var1', 'f4'), ('var2', 'f4'), ('instrument', 'f4'),
                                             ('str_instr', 'a10')])

        structdes['var1'] = self.des[:, 0][:, None]
        structdes['var2'] = self.des[:, 1][:, None]
        structdes['instrument'] = self.instr[:, None]

        string_var = [stringabc[0:5], stringabc[5:10], stringabc[10:15], stringabc[15:20], stringabc[20:25]]
        string_var *= 5
        self.string_var = np.array(sorted(string_var))

        structdes['str_instr'] = self.string_var[:, None]
        self.structdes = structdes
        self.recdes = structdes.view(np.recarray)


class TestCategoricalNumerical(TestCategorical):
    # TODO: use assert_raises to check that bad inputs are taken care of

    def test_array2d(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = to_categorical(des, col=2)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 10)

    def test_array1d(self):
        des = to_categorical(self.instr)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 6)

    def test_array2d_drop(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = to_categorical(des, col=2, drop=True)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 9)

    def test_array1d_drop(self):
        des = to_categorical(self.instr, drop=True)
        assert_array_equal(des, self.dummy)
        assert_equal(des.shape[1], 5)

    def test_recarray2d(self):
        des = to_categorical(self.recdes, col='instrument')
        # better way to do this?
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray2dint(self):
        des = to_categorical(self.recdes, col=2)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray1d(self):
        instr = self.structdes['instrument'].view(np.recarray)
        dum = to_categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_recarray1d_drop(self):
        instr = self.structdes['instrument'].view(np.recarray)
        dum = to_categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    def test_recarray2d_drop(self):
        des = to_categorical(self.recdes, col='instrument', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray2d(self):
        des = to_categorical(self.structdes, col='instrument')
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray2dint(self):
        des = to_categorical(self.structdes, col=2)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray1d(self):
        instr = self.structdes['instrument'].view(dtype=[('var1', 'f4')])
        dum = to_categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_structarray2d_drop(self):
        des = to_categorical(self.structdes, col='instrument', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray1d_drop(self):
        instr = self.structdes['instrument'].view(dtype=[('var1', 'f4')])
        dum = to_categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    def test_arraylike2d(self):
        des = to_categorical(self.structdes.tolist(), col=2)
        des = np.array(des)
        test_des = des[:, -5:].astype(float)
        assert_array_equal(test_des, self.dummy)
        assert_equal(des.shape[1], 9)

    def test_arraylike1d(self):
        instr = self.structdes['instrument'].tolist()
        dum = to_categorical(instr)
        dum = np.array(dum)
        test_dum = dum[:, -5:].astype(float)
        assert_array_equal(test_dum, self.dummy)
        assert_equal(dum.shape[1], 6)

    def test_arraylike2d_drop(self):
        des = to_categorical(self.structdes.tolist(), col=2, drop=True)
        des = np.array(des)
        test_des = des[:, -5:].astype(float)
        assert_array_equal(test_des, self.dummy)
        assert_equal(des.shape[1], 8)

    def test_arraylike1d_drop(self):
        instr = self.structdes['instrument'].tolist()
        dum = to_categorical(instr, drop=True)
        dum = np.array(dum)
        assert_array_equal(dum, self.dummy)
        assert_equal(dum.shape[1], 5)


class TestCategoricalString(TestCategorical):

    def test_array2d(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = to_categorical(des, col=2)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 10)

    def test_array1d(self):
        des = to_categorical(self.instr)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 6)

    def test_array2d_drop(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = to_categorical(des, col=2, drop=True)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 9)

    def test_array1d_drop(self):
        des = to_categorical(self.string_var, drop=True)
        assert_array_equal(des, self.dummy)
        assert_equal(des.shape[1], 5)

    def test_recarray2d(self):
        des = to_categorical(self.recdes, col='str_instr')
        # better way to do this?
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray2dint(self):
        des = to_categorical(self.recdes, col=3)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_recarray1d(self):
        instr = self.structdes['str_instr'].view(np.recarray)
        dum = to_categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_recarray1d_drop(self):
        instr = self.structdes['str_instr'].view(np.recarray)
        dum = to_categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    def test_recarray2d_drop(self):
        des = to_categorical(self.recdes, col='str_instr', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray2d(self):
        des = to_categorical(self.structdes, col='str_instr')
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray2dint(self):
        des = to_categorical(self.structdes, col=3)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 9)

    def test_structarray1d(self):
        instr = self.structdes['str_instr'].view(dtype=[('var1', 'a10')])
        dum = to_categorical(instr)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names[-5:]]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 6)

    def test_structarray2d_drop(self):
        des = to_categorical(self.structdes, col='str_instr', drop=True)
        test_des = np.column_stack(([des[_] for _ in des.dtype.names[-5:]]))
        assert_array_equal(test_des, self.dummy)
        assert_equal(len(des.dtype.names), 8)

    def test_structarray1d_drop(self):
        instr = self.structdes['str_instr'].view(dtype=[('var1', 'a10')])
        dum = to_categorical(instr, drop=True)
        test_dum = np.column_stack(([dum[_] for _ in dum.dtype.names]))
        assert_array_equal(test_dum, self.dummy)
        assert_equal(len(dum.dtype.names), 5)

    def test_arraylike2d(self):
        des = np.column_stack((self.des, self.instr, self.des)).tolist()
        des = to_categorical(des, col=2)
        des = np.array(des)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 10)

    def test_arraylike1d(self):
        des = to_categorical(self.instr.tolist())
        des = np.array(des)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 6)

    def test_arraylike2d_drop(self):
        des = np.column_stack((self.des, self.instr, self.des))
        des = to_categorical(des.tolist(), col=2, drop=True)
        des = np.array(des)
        assert_array_equal(des[:, -5:], self.dummy)
        assert_equal(des.shape[1], 9)

    def test_arraylike1d_drop(self):
        des = to_categorical(self.string_var.tolist(), drop=True)
        des = np.array(des)
        assert_array_equal(des, self.dummy)
        assert_equal(des.shape[1], 5)
