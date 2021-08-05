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

