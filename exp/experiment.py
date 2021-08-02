import runtime
import mtx
import numpy as np
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 128
ALL_RUNTIME = {
    "pytorch":runtime.pytorch_runtime,
    "pytorch_sparse":runtime.pytorch_sparse_runtime,
    "tensorflow":runtime.tensorflow_runtime,
    "tensorflow_sparse":runtime.tensorflow_sparse_runtime,
    "sgk_sparse":runtime.sgk_sparse_runtime
}

def experiment(arg):
    batch_size = BATCH_SIZE

    if(not os.path.isdir(arg.output_path)):
        os.mkdir(arg.output_path)
    for subdir, dirs, files in os.walk(arg.input_path):
        if(len(dirs) == 0): 
            dimension = subdir.split("/")[-1].strip()
            print("Experiment on matrix size {}".format(dimension))
            sub_res = {}
            for f in tqdm(files):
                f_path = os.path.join(subdir,f)
                sparsity = float(f.split(".")[0])
                A = mtx.load_from_mtx(f_path)
                B =  np.random.randn(A.shape[1], batch_size).astype(np.float32)
                for name, func in ALL_RUNTIME.items():
                    res = func(A,B)
                    sub_res.setdefault(name,{})[sparsity]=res
            print("Generating graphs")    
            plt.figure(figsize=(16,9))
            for name,data in sub_res.items():
                plt.plot(list(data.keys()),list(data.values()),label=name)
            plt.title("Sparse Kernel | Matrix Size: {}".format(dimension))
            plt.legend()
            plt.savefig(os.path.join(arg.output_path,"{}.png".format(dimension)))
            print("*"*40)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help = "Data path",type=str, required=True)
    parser.add_argument("-o", "--output_path", help = "Output path",type=str, required=True)
    arg = parser.parse_args()
    experiment(arg)
