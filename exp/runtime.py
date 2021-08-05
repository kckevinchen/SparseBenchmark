import time

# Must set variable prior to tensorflow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.compat.v1 as tf1

# Must load mkl prior to pytorch
from mkl_helper.mkl_util import time_mkl_sparse_dense_matmul


import torch
import torch.utils.benchmark as benchmark
import numpy as np
from scipy import sparse

from _c.cpp_lib import *
from sgk.sparse import ops
from sgk.sparse import sparse_matrix



# Disable TF2.
tf1.disable_v2_behavior()

REPS = 100
MKL_REPS = 10
BURN_ITERS = 10



def tensorflow_runtime(A, B, reps=REPS, burn_iters=BURN_ITERS):
    """
    Given the sparse matrix A and dense matrix B return the runtime of the
    tf.matmul function
    """
    with tf.device("/GPU:0"):
        lhs = tf.constant(A)
        rhs = tf.constant(B)
    times = []

    for _ in range(burn_iters):
        tf.matmul(lhs, rhs)

    for _ in range(reps):
        start = time.perf_counter()
        tf.raw_ops.MatMul(a=lhs, b=rhs)
        end = time.perf_counter()
        times.append((end - start)*1000.0)
    del lhs
    del rhs
    return np.median(times)


def tensorflow_sparse_runtime(A, B, reps=REPS, burn_iters=BURN_ITERS):
    """
    Given the sparse matrix A and dense matrix B return the runtime of the
    tf.sparse.sparse_dense_matmul function
    """
    with tf.device("/GPU:0"):
        lhs = tf.sparse.from_dense(A)
        rhs = tf.constant(B)
    times = []
    for _ in range(burn_iters):
        tf.sparse.sparse_dense_matmul(lhs, rhs)

    for _ in range(reps):
        start = time.perf_counter()
        tf.sparse.sparse_dense_matmul(lhs, rhs)
        end = time.perf_counter()
        times.append((end - start)*1000.0)
    del lhs
    del rhs
    return np.median(times)


def pytorch_runtime(A, B, reps=REPS, burn_iters=BURN_ITERS):
    """
    Given the sparse matrix A and dense matrix B return the runtime of the
    torch.mm function
    """
    device = 'cuda'
    lhs = torch.from_numpy(A).t().to(device)
    rhs = torch.from_numpy(B).t().to(device)

    # t = benchmark.Timer(
    #     stmt='torch.mm(lhs, rhs)',
    #     globals={'lhs': rhs, 'rhs': lhs}).blocked_autorange()
    # result = t.median*1000
    times = []
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(burn_iters):
        torch.matmul(lhs, rhs)
    for _ in range(reps):
        start.record()
        torch.matmul(lhs, rhs)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))# millisecs
    del lhs
    del rhs
    return np.median(times)


def pytorch_sparse_runtime(A, B, reps=REPS, burn_iters=BURN_ITERS):
    """
    Given the sparse matrix A and dense matrix B return the runtime of the
    torch.sparse.mm function
    """
    device = 'cuda'
    A = sparse.coo_matrix(A)
    values = A.data
    indices = np.vstack((A.row, A.col))

    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = A.shape

    lhs = torch.sparse_coo_tensor(
        i, v, torch.Size(shape)).coalesce().to(device)
    rhs = torch.from_numpy(B).to(device=device)

    t = benchmark.Timer(
        stmt='torch.sparse.mm(lhs, rhs)',
        globals={'lhs': lhs, 'rhs': rhs}).blocked_autorange()
    result = t.median*1000
    del lhs
    del rhs
    return result


def sgk_sparse_runtime(A, B, reps=REPS, burn_iters=BURN_ITERS):
    """
    Given the sparse matrix A and dense matrix B return the runtime of the
    sgk.spmm function
    https://arxiv.org/abs/2006.10901
    """
    tf1.reset_default_graph()
    lhs = sparse_matrix.SparseMatrix(
        "lhs", matrix=A)
    rhs = tf1.constant(B, dtype=tf1.float32)
    output = ops.spmm(lhs, rhs)

    times = []
    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        for _ in range(burn_iters):
            sess.run(output)

        for _ in range(reps):
            start = time.perf_counter()
            sess.run(output)
            end = time.perf_counter()
        times.append((end - start)*1000.0)
    del lhs
    del rhs

    return np.median(times)


def cublas_runtime(A, B, reps=REPS, burn_iters=BURN_ITERS):
    """
    Given the sparse matrix A and dense matrix B return the runtime of the 
    cublas code (cpp code binded in python)
    """
    m, k = A.shape
    n = B.shape[1]
    times = []

    for _ in range(burn_iters):
        cuBLAS(m, n, k, A.reshape(-1), B.reshape(-1))

    for _ in range(reps):
        times.append(cuBLAS(m, n, k, A.reshape(-1), B.reshape(-1)))
    return np.median(times)*1e-6


def cusparse_runtime(A, B, reps=REPS, burn_iters=BURN_ITERS):
    """
    Given the sparse matrix A and dense matrix B return the runtime of the 
    cusparse code (cpp code binded in python)
    """
    sparse_A = sparse.csr_matrix(A)
    m, k = A.shape
    n = B.shape[1]
    times = []
    for _ in range(burn_iters):
        cuSPARSE(m, n, k, sparse_A.nnz, sparse_A.indptr,
                 sparse_A.indices, sparse_A.data, B.flatten())

    for _ in range(reps):
        times.append(cuSPARSE(m, n, k, sparse_A.nnz, sparse_A.indptr,
                     sparse_A.indices, sparse_A.data, B.flatten()))
    return np.median(times)*1e-6


def _dense_to_sparse(matrix):
    """Converts dense numpy matrix to a csr sparse matrix."""
    assert len(matrix.shape) == 2

    # Extract the nonzero values.
    values = matrix.compress((matrix != 0).flatten())

    # Calculate the offset of each row.
    mask = (matrix != 0).astype(np.int32)
    row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))),
                                 axis=0)
    # Create the row indices and sort them.
    row_indices = np.argsort(-1 * np.diff(row_offsets))
    # Extract the column indices for the nonzero values.
    x = mask * (np.arange(matrix.shape[1]) + 1)
    column_indices = x.compress((x != 0).flatten())
    column_indices = column_indices - 1

    # Cast the desired precision.
    values = values.astype(np.float32)
    row_indices, row_offsets, column_indices = [
        x.astype(np.uint32) for x in
        [row_indices, row_offsets, column_indices]
    ]
    return values, row_indices, row_offsets, column_indices


def sgk_op_runtime(A, B, reps=REPS, burn_iters=BURN_ITERS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the 
	sgk code (cpp code binded in python)
	"""
	m, k = A.shape
	n = B.shape[1]
	values, row_indices, row_offsets, column_indices = _dense_to_sparse(A)
	nonzeros = values.size
	times = []
	for _ in range(burn_iters):
		sgkSPARSE(m, n, k, nonzeros, values, row_indices,
					row_offsets, column_indices, B.flatten())
	for _ in range(reps):
		times.append(sgkSPARSE(m, n, k, nonzeros, values, row_indices,
								row_offsets, column_indices, B.flatten()))
	return np.median(times)*1e-6


def mkl_runtime(A, B, reps=MKL_REPS, burn_iters=BURN_ITERS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the 
	sgk code (cpp code binded in python)
	"""
	A = sparse.csr_matrix(A)
	times = []

	for _ in range(reps):
		times.append(time_mkl_sparse_dense_matmul(A,B))
	return np.median(times)

if __name__ == "__main__":

    def gen_test_input(m, n, k, sparsity):
        """
        return A as a sparse matrix and B as a dense
        A is of shape (m, k)
        B is of shape (k, n)
        """
        sparse_A = sparse.random(
            m, k, 1 - sparsity, format='csr', dtype=np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        return sparse_A, B
    m = 512
    n = 1024
    k = 512

    sparsity = 0.4
    A, B = gen_test_input(m, n, k, sparsity)
    print(mkl_runtime(A, B))
    print(pytorch_runtime(A.toarray(), B))   
