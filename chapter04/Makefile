# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -I../common

# Targets
TARGETS = julia_gpu julia_cpu

# Linked file
STB_IMAGE_FILE = ../common/stb/stb_image_write.cpp

all: $(TARGETS)

# GPU version
julia_gpu: julia_gpu.cu $(STB_IMAGE_FILE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# CPU version
julia_cpu: julia_cpu.cu $(STB_IMAGE_FILE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Clean up
clean:
	rm -f $(TARGETS)
