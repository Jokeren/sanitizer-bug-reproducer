EXEC := main

CXX := nvcc

GPU_FLAGS += -O3 -lineinfo -arch sm_70

CUBIN_FLAGS += -cubin

CUDA_PATH := /usr/local/cuda

LDFLAGS += -L../../ -ltool -L$(CUDA_PATH)/extras/Sanitizer -lsanitizer-public

CXXFLAGS := $(GPU_FLAGS)

all: exec
	
exec: $(EXEC)

$(EXEC): % : %.cu $(DEVICE_OBJ)
	$(CXX) -o $@ $^ $(SHOWFLAGS) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -rf $(EXEC) *.o *.dot *.hpcstruct *.cubin

print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true
