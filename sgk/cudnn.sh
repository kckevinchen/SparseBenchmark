OS="ubuntu2004"
cudnn_version="8.2.2.26"
cuda_version="cuda11.4"
wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin 

mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
apt-get update
apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}