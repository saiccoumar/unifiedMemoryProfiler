NVCC     := nvcc
CXXFLAGS := -O3 -std=c++14

# Pass TARGET=4090 or TARGET=thor (default: thor)
TARGET ?= thor

BINS := discrete_benchmark_$(TARGET) unified_benchmark_$(TARGET)

all: $(BINS)

%_$(TARGET): %_$(TARGET).cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f discrete_benchmark_4090 discrete_benchmark_thor \
	      unified_benchmark_4090 unified_benchmark_thor *.csv

.PHONY: all clean
