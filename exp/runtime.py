import time
import numpy as np
from scipy import sparse

import torch
import torch.utils.benchmark as benchmark

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow.compat.v1 as tf1
import tensorflow as tf

from sgk.sparse import sparse_matrix
from sgk.sparse import ops


# Disable TF2.
tf1.disable_v2_behavior()

REPS = 100

BURN_ITERS = 10

def tensorflow_runtime(A, B, reps=REPS,burn_iters=BURN_ITERS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the
	tf.matmul function
	"""
	with tf.device("/GPU:0"):
		lhs = tf.constant(A)
		rhs = tf.constant(B)
	print(type(lhs))
	print(lhs.device)
	times = []

	for _ in range(burn_iters):
		tf.matmul(lhs, rhs)

	for _ in range(reps):
		start = time.perf_counter()
		tf.matmul(lhs, rhs)
		end = time.perf_counter()
		times.append((end - start)*1000.0)
	del lhs
	del rhs
	return np.mean(times)


def tensorflow_sparse_runtime(A, B, reps=REPS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the
	tf.sparse.sparse_dense_matmul function
	"""
	with tf.device("/GPU:0"):
		lhs = tf.sparse.from_dense(A)
		rhs = tf.constant(B)
	times = []
	for _ in range(reps):
		start = time.perf_counter()
		tf.sparse.sparse_dense_matmul(lhs, rhs)
		end = time.perf_counter()
		times.append((end - start)*1000.0)
	del lhs
	del rhs
	return np.median(times)


def pytorch_runtime(A, B, reps=REPS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the
	torch.mm function
	"""
	device = 'cuda'
	lhs = torch.from_numpy(A).t().to(device=device)
	rhs = torch.from_numpy(B).t().to(device=device)

	
	t = benchmark.Timer(
    stmt='torch.mm(lhs, rhs)',
    globals={'lhs': rhs,'rhs':lhs})
	result = t.timeit(REPS).median*1000
	del lhs
	del rhs
	return result


def pytorch_sparse_runtime(A, B, reps=REPS):
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
		i, v, torch.Size(shape)).to(device).coalesce()
	rhs = torch.from_numpy(B).to(device=device)

	t = benchmark.Timer(
    stmt='torch.sparse.mm(lhs, rhs)',
    globals={'lhs': lhs,'rhs':rhs})
	result = t.timeit(REPS).median*1000
	del lhs
	del rhs
	return result


def sgk_sparse_runtime(A, B, reps=REPS,burn_iters= BURN_ITERS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the
	sgk.spmm function
	https://arxiv.org/abs/2006.10901
	"""
	lhs = sparse_matrix.SparseMatrix(
		"lhs_{}".format(round(time.time() * 1000)), matrix=np.array(A))
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
