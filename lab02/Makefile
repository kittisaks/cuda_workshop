TARGET_0 = tb
SRC_0    = tid_basic.cu
TARGET_1 = tf
SRC_1    = tid_full.cu

include ../Makefile.common

all: $(TARGET_0) $(TARGET_1)
	@printf "\n"

clean:
	$(RM) $(RM_FLAGS) $(TARGET_0)
	$(RM) $(RM_FLAGS) $(TARGET_1)

$(TARGET_0) : $(SRC_0)
	$(NVCC) $(NVCC_BFLAGS) $(BEG_X_FLAGS) $(CPPNV_BFLAGS) $(END_X_FLAGS) $^ -o $@
	@printf "Thread Identification (Full) Example Built.\n"

$(TARGET_1) : $(SRC_1)
	$(NVCC) $(NVCC_BFLAGS) $(BEG_X_FLAGS) $(CPPNV_BFLAGS) $(END_X_FLAGS) $^ -o $@
	@printf "Thread Identification (Basic) Example Built.\n"

