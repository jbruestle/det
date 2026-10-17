NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= -gencode arch=compute_120,code=sm_120 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_80,code=sm_80
# -fmad=false: never contract into FMA.  No fast-math.  Defaults keep prec-div/prec-sqrt=true, ftz=false.
NVFLAGS = -O3 -std=c++17 -fmad=false -Xcompiler -fPIC,-ffp-contract=off -lineinfo $(ARCH)
SRCS = norm.cu gemm.cu attn.cu gdn.cu sample.cu
OBJS = $(SRCS:.cu=.o)
all: libdet.so
%.o: %.cu common.cuh det.h
	$(NVCC) $(NVFLAGS) -c $< -o $@
libdet.so: $(OBJS)
	$(NVCC) $(NVFLAGS) -shared $(OBJS) -o $@
clean:
	rm -f *.o libdet.so
# per-kernel test libraries (so parallel work on other files cannot break your build)
libdet_gemm.so: norm.o gemm.o
	$(NVCC) $(NVFLAGS) -shared $^ -o $@
libdet_attn.so: norm.o attn.o
	$(NVCC) $(NVFLAGS) -shared $^ -o $@
libdet_gdn.so: norm.o gdn.o
	$(NVCC) $(NVFLAGS) -shared $^ -o $@
libdet_glue.so: norm.o
	$(NVCC) $(NVFLAGS) -shared $^ -o $@
libdet_sample.so: norm.o sample.o
	$(NVCC) $(NVFLAGS) -shared $^ -o $@
