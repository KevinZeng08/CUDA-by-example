CONFIG_LOCAL := ./config/Makefile.config

include $(CONFIG_LOCAL)

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -I../common

ifeq ($(DEBUG), 1)
	NVCC_FLAGS += -g -G
else
	NVCC_FLAGS += -O3
endif

# Targets
TARGETS = ray_noconst ray_dynamic ray_noconst_dynamic ray 

# Linked file
STB_IMAGE_FILE = ../common/stb/stb_image_write.cpp

all: $(TARGETS)

# Dynamic Ray Tracing
ray_noconst_dynamic: ray_noconst_dynamic.cu $(STB_IMAGE_FILE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Dynamic Ray Tracing
ray_dynamic: ray_dynamic.cu $(STB_IMAGE_FILE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# GPU version
ray_noconst: ray_noconst.cu $(STB_IMAGE_FILE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Constant memory version
ray: ray.cu $(STB_IMAGE_FILE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Clean up
clean:
	rm -f $(TARGETS)
