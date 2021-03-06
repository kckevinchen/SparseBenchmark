import runtime
import mtx
import numpy as np
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import make_interp_spline, BSpline

BATCH_SIZE = 128

SMOOTH = 2
SMOOTH_INT = 100
ALL_RUNTIME = {
    "pytorch":runtime.pytorch_runtime,
    "pytorch_sparse":runtime.pytorch_sparse_runtime,
    "tensorflow":runtime.tensorflow_runtime,
    "tensorflow_sparse":runtime.tensorflow_sparse_runtime,
    "sgk_tf":runtime.sgk_sparse_runtime,
    "cuBLAS":runtime.cublas_runtime,
    "cuSPARSE":runtime.cusparse_runtime,
    "sgk_op" :runtime.sgk_op_runtime,
    "mkl": runtime.mkl_runtime
}

ALL_KERNEL = {
    "cuBLAS":runtime.cublas_runtime,
    "cuSPARSE":runtime.cusparse_runtime,
    "sgk_op" :runtime.sgk_op_runtime,
}
def generate_line(keys,values,name):
    keys = np.array(list(keys))
    values = np.array(list(values))
    idx = np.argsort(keys)
    x = keys[idx]
    y = values[idx]
    plt.scatter(x, y, alpha=0.5,s=15,label = name)

    total_len = np.prod(x.shape)
    xnew = np.linspace(x.min(), x.max(), round(total_len*SMOOTH)) 
    # xnew = np.linspace(x.min(), x.max(), SMOOTH_INT) 

    spl = make_interp_spline(x, y, k=3)  # type: BSpline
    power_smooth = spl(xnew)

    plt.plot(xnew, power_smooth,label = name)

def experiment(arg):
    batch_size = BATCH_SIZE
    if(arg.raw_kernel):
        runtimes = ALL_KERNEL
    else:
        runtimes = ALL_RUNTIME
    if(not os.path.isdir(arg.output_path)):
        os.mkdir(arg.output_path)
    for subdir, dirs, files in os.walk(arg.input_path):
        if(len(dirs) == 0): 
            dimension = subdir.split("/")[-1].strip()
            print("Experiment on matrix size {}".format(dimension))
            sub_res = {}
            for f in tqdm(files):
                f_path = os.path.join(subdir,f)
                sparsity = 1 - float(".".join(f.split(".")[:-1]))
                A = mtx.load_from_mtx(f_path)
                B =  np.random.randn(A.shape[1], batch_size).astype(np.float32)
                for name, func in runtimes.items():
                    res = func(A,B)
                    sub_res.setdefault(name,{})[sparsity]=res
            print("Generating graphs")    
            plt.figure(figsize=(16,9))
            for name,data in sub_res.items():
                generate_line(data.keys(),data.values(),name)
                # keys = np.array(list(data.keys()))
                # values = np.array(list(data.values()))
                # idx = np.argsort(keys)
                # plt.plot(keys[idx],values[idx],label=name)
            plt.title("Sparse Kernel | Matrix Size: {}".format(dimension))
            plt.legend()
            plt.xlabel("Sparsity")
            plt.ylabel("Runtime (ms)")
            plt.savefig(os.path.join(arg.output_path,"{}.png".format(dimension)))
            plt.close()
            print("*"*40)
            del sub_res


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help = "Data path",type=str, required=True)
    parser.add_argument("-o", "--output_path", help = "Output path",type=str, required=True)
    parser.add_argument("-k", "--raw_kernel", help = "Output raw kernel",action='store_true')
    arg = parser.parse_args()
    experiment(arg)
