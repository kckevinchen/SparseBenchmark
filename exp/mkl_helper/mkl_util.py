
from mkl_helper._mkl_interface import (MKL, _sanity_check, _empty_output_check, _type_check, _create_mkl_sparse,
                                           _destroy_mkl_handle, matrix_descr, debug_print, _convert_to_csr,
                                           _get_numpy_layout, _check_return_value, LAYOUT_CODE_C, LAYOUT_CODE_F,
                                           _out_matrix)
import numpy as np
import ctypes as _ctypes
import scipy.sparse as _spsparse
import time
# June 2nd 2016 version.

def time_mkl_sparse_dense_matmul(matrix_a, matrix_b, scalar=1., transpose=False, out=None, out_scalar=None, out_t=None):
    """
    Time mkl multiply a sparse and a dense matrix
    mkl_sparse_?_mm requires the left (A) matrix to be sparse and the right (B) matrix to be dense
    This requires conversion of the sparse matrix to CSR format for some dense arrays.
    A must be CSR if B is column-major. Otherwise CSR or CSC are acceptable.
    :param matrix_a: Left (A) matrix
    :type matrix_a: sp.spmatrix.csr, sp.spmatrix.csc
    :param matrix_b: Right (B) matrix
    :type matrix_b: np.ndarray
    :param scalar: A value to multiply the result matrix by. Defaults to 1.
    :type scalar: float
    :param transpose: Return AT (dot) B instead of A (dot) B.
    :type transpose: bool
    :param out: Add the dot product to this array if provided.
    :type out: np.ndarray, None
    :param out_scalar: Multiply the out array by this scalar if provided.
    :type out_scalar: float, None
    :return: A (dot) B as a dense array in either column-major or row-major format
    :rtype: np.ndarray
    """
    _mkl_handles = []

    output_shape = (matrix_a.shape[1] if transpose else matrix_a.shape[0], matrix_b.shape[1])
    layout_b, ld_b = _get_numpy_layout(matrix_b, second_arr=out)

    try:
        # Prep MKL handles and check that matrixes are compatible types
        # MKL requires CSR format if the dense array is column-major
        if layout_b == LAYOUT_CODE_F and not _spsparse.isspmatrix_csr(matrix_a):
            mkl_non_csr, dbl = _create_mkl_sparse(matrix_a)
            _mkl_handles.append(mkl_non_csr)
            mkl_a = _convert_to_csr(mkl_non_csr)
        else:
            mkl_a, dbl = _create_mkl_sparse(matrix_a)

        _mkl_handles.append(mkl_a)

        # Set functions and types for float or doubles
        output_ctype = _ctypes.c_double if dbl else _ctypes.c_float
        output_dtype = np.float64 if dbl else np.float32
        func = MKL._mkl_sparse_d_mm if dbl else MKL._mkl_sparse_s_mm

        # Allocate an output array
        output_arr = _out_matrix(output_shape, output_dtype, order="C" if layout_b == LAYOUT_CODE_C else "F",
                                out_arr=out, out_t=out_t)

        output_layout, output_ld = _get_numpy_layout(output_arr, second_arr=matrix_b)

        # mkl call
        start = time.perf_counter()
        ret_val = func(11 if transpose else 10,
                    scalar,
                    mkl_a,
                    matrix_descr(),
                    layout_b,
                    matrix_b,
                    output_shape[1],
                    ld_b,
                    float(out_scalar) if out_scalar is not None else 1.,
                    output_arr.ctypes.data_as(_ctypes.POINTER(output_ctype)),
                    output_ld)
        end = time.perf_counter()
        # Check return
        del output_arr
        _check_return_value(ret_val, func.__name__)
        return (end - start)*1000.0

    finally:
        for _mhandle in _mkl_handles:
            _destroy_mkl_handle(_mhandle)

if __name__ == "__main__":
    def gen_test_input(m, n, k, sparsity):
        """
        return A as a sparse matrix and B as a dense
        A is of shape (m, k)
        B is of shape (k, n)
        """
        sparse_A = _spsparse.random(
            m, k, 1 - sparsity, format='csr', dtype=np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        return sparse_A, B
    m = 512
    n = 1024
    k = 512

    sparsity = 0.4
    A, B = gen_test_input(m, n, k, sparsity)
    print(time_mkl_sparse_dense_matmul(A,B))
