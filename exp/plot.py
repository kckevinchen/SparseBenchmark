import numpy as np
from scipy import sparse
from _c.cpp_lib import *
import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import time
from scipy.stats.mstats import gmean
from tqdm import tqdm

REPS = 10

def tensorflow_runtime(A, B, reps=REPS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the 
	tf.sparse.sparse_dense_matmul function
	"""
	A = A.A
	t_s_A = tf.sparse.from_dense(A)
	times = []
	for _ in range(reps):
		start = time.perf_counter()
		tf.sparse.sparse_dense_matmul(t_s_A, B)
		end = time.perf_counter()
		times.append(end - start)
	return 1000.0 * np.mean(times)

def cublas_runtime(A, B, reps=REPS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the 
	cublas code (cpp code binded in python)
	"""
	A = A.A
	m, k = A.shape
	n = B.shape[1]
	dense_time = []
	for _ in range(reps):
		dense_time.append(cuBLAS(m, n, k, A.reshape(-1),B.reshape(-1)))
	return np.mean(dense_time)

def cusparse_runtime(A, B, reps=REPS):
	"""
	Given the sparse matrix A and dense matrix B return the runtime of the 
	cusparse code (cpp code binded in python)
	"""
	sparse_A = A
	m, k = A.A.shape
	n = B.shape[1]
	sparse_time = []
	for _ in range(reps):
		sparse_time.append(cuSPARSE(m,n,k, sparse_A.nnz, sparse_A.indptr, sparse_A.indices, sparse_A.data, B.flatten()))
	return np.mean(sparse_time)



def gen_test_input(m,n,k,sparsity):
	"""
	return A as a sparse matrix and B as a dense
	A is of shape (m, k)
	B is of shape (k, n)
	"""
	sparse_A = sparse.random(m, k, 1 - sparsity, format = 'csr', dtype=np.float32)
	B = np.random.randn(k, n).astype(np.float32)
	return sparse_A, B

def experiment(sparsity = 0.9):
	#first experiment: square mtx matmul
	dense_time, sparse_time, tf_time = [], [], []
	# for x in tqdm(np.linspace(10, 1 << 11, num=10, dtype=np.int32), leave=False):
	for x in np.linspace(10, 1<<11, num=10, dtype=np.int32):
		A, B = gen_test_input(x,x,x,sparsity) 
		dense_time.append(cublas_runtime(A,B))
		sparse_time.append(cusparse_runtime(A, B))
		tf_time.append(tensorflow_runtime(A, B))
	df = pd.DataFrame({"cuBLAS": dense_time, "cuSPARSE": sparse_time, "tensorflow": tf_time}, index=list(np.linspace(10, 1 << 13, num=10, dtype=np.int32)))
	df.plot.line()	
	plt.title("cuSparse -- cuBlas compare | sparsity = {}".format(sparsity))
	plt.xlabel("squared matrix dimension")
	plt.ylabel("run time (ms)")
	plt.savefig("plots/{}_sparsity.png".format(sparsity))	

def plot():
	for s in tqdm([0.9, 0.95, 0.99, 0.999], leave=False):
		experiment(s)

if __name__ == "__main__":
	plot()