CUDNN_PATH := cudnn
INCLUDES := `python -m pybind11 --includes` -I $(CUDNN_PATH)/include
NVCC := /usr/local/cuda-11/bin/nvcc
CUDA_TOOLKIT := /usr/local/cuda-11/
NVCC_INC := -I/usr/local/cuda-11/include
LIBS         := -lcudart -lcusparse -lcublas
CXX_FLAGS := -std=c++11 -g -O2
LIBRARY_DIR := cpp_lib.cu

all:
	${NVCC} ${NVCC_INC} $(CXX_FLAGS) $(LIBRARY_DIR) -shared -Xcompiler -fPIC $(INCLUDES) -o cpp_lib`python3-config --extension-suffix` $(LIBS)

clean:
	rm *.so
	rm *.o
