
GEN_TARGET = vaddgen
CPU_TARGET = vaddcpu
GPU_TARGET = vaddgpu

GEN_SRC = vaddgen.cpp
CPU_SRC = vaddcpu.cpp
GPU_SRC = vaddgpu.cu

########## DO NOT MODIFIED BELOW; UNLESS YOU KNOW WHAT YOU ARE DOING ##########

include ../Makefile.common

OBJ_DIR = obj
GEN_OBJ = $(addprefix $(OBJ_DIR)/, $(GEN_SRC:.cpp=.o))
CPU_OBJ = $(addprefix $(OBJ_DIR)/, $(CPU_SRC:.cpp=.o))
GPU_OBJ := $(GPU_SRC:.cpp=.o)
GPU_OBJ := $(addprefix $(OBJ_DIR)/, $(GPU_OBJ:.cu=.o))

all: $(CPU_TARGET) $(GPU_TARGET)
	@printf "Build Succesful\n"

clean:
	$(RM) $(RM_FLAGS) $(GEN_OBJ) $(GEN_TARGET)
	$(RM) $(RM_FLAGS) $(CPU_OBJ) $(CPU_TARGET)
	$(RM) $(RM_FLAGS) $(GPU_OBJ) $(GPU_TARGET)
	$(RM) $(RM_FLAGS) $(OBJ_DIR)/*.d

include $(wildcard $(OBJ_DIR)/*.d)

$(GEN_TARGET): $(GEN_OBJ)
	$(LD) $^ -o $@ -lgomp

$(CPU_TARGET): $(CPU_OBJ)
	$(LD) $^ -o $@ -lgomp

$(GPU_TARGET): $(GPU_OBJ)
	$(CULD) $^ -o  $@

$(OBJ_DIR)/%.o:%.cpp
	$(CPP) $(INC_COMMON) -fopenmp $(CPP_FLAGS) $< -o $@

$(OBJ_DIR)/%.o:%.cu
	$(NVCC) $(INC_COMMON) $(NVCC_FLAGS) $(BEG_X_FLAGS) $(CPPNV_FLAGS) $(END_X_FLAGS) $< -o $@



