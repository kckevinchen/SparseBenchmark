# SparseBenchmark

## Setup
```
sudo docker build . -t sparsebenchmark
sudo docker run --runtime=nvidia -v {path/to/sparsebenchmark/exp}:/mount/exp -it sparsebenchmark

#install MKL
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O - | apt-key add -
echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
apt install intel-mkl

#add mkl to LD_LIBRARY_PATH (whereis mkl)
export LD_LIBRARY_PATH={path/to/mkl}:$LD_LIBRARY_PATH
```

## Usage
```
cd exp
python experiment.py -i {path/to/data} -p {output_path}
```

