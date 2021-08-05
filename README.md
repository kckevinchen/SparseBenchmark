# SparseBenchmark

## Setup
```
sudo docker build . -t sparsebenchmark
sudo docker run --runtime=nvidia -v {path/to/sparsebenchmark/exp}:/mount/exp -it sparsebenchmark
```

## Usage
```
cd exp
python experiment.py -i {path/to/data} -p {output_path}
```

The MKL files are adpoted from sparse_dot_mkl(https://github.com/flatironinstitute/sparse_dot)

The cuSPARSE&cuBLAS are adopted from Tian(https://github.com/Tian99Yu/NNW_RT/)
