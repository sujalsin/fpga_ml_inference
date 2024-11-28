CXX := g++
CXXFLAGS := -std=c++14 -I$(XILINX_HLS)/include
LDFLAGS :=

SRC_DIR := neural_net
TEST_DIR := test
BUILD_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJS := $(TEST_SRCS:%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean test synthesis export

all: test synthesis export

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(OBJS) $(TEST_OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(OBJS) $(TEST_OBJS) $(LDFLAGS) -o $(BUILD_DIR)/test_neural_net
	./$(BUILD_DIR)/test_neural_net

synthesis:
	vivado_hls -f scripts/synthesize.tcl

export:
	vivado_hls -f scripts/export.tcl

clean:
	rm -rf $(BUILD_DIR)
	rm -rf *.log
