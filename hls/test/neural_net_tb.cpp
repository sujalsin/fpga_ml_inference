#include "../neural_net/neural_net.hpp"
#include <iostream>
#include <cmath>

// Test data generation
void generate_test_data(
    fixed_t weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE],
    fixed_t bias[MAX_LAYER_SIZE],
    int input_size,
    int output_size
) {
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            weights[i][j] = fixed_t(0.1) * (i + j);
        }
        bias[i] = fixed_t(0.01) * i;
    }
}

int main() {
    // Create test streams
    hls::stream<axi_data_t> input_stream("input");
    hls::stream<axi_data_t> output_stream("output");
    
    // Create test weights and biases
    fixed_t layer1_weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE];
    fixed_t layer1_bias[MAX_LAYER_SIZE];
    fixed_t layer2_weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE];
    fixed_t layer2_bias[MAX_LAYER_SIZE];
    
    // Generate test data
    generate_test_data(layer1_weights, layer1_bias, MAX_INPUT_SIZE, MAX_LAYER_SIZE);
    generate_test_data(layer2_weights, layer2_bias, MAX_LAYER_SIZE, MAX_OUTPUT_SIZE);
    
    // Generate input data
    for (int i = 0; i < MAX_INPUT_SIZE; i++) {
        axi_data_t input_data;
        input_data.data = i;
        input_data.keep = 1;
        input_data.strb = 1;
        input_data.last = (i == MAX_INPUT_SIZE-1);
        input_stream.write(input_data);
    }
    
    // Run inference
    neural_net_inference(
        input_stream,
        output_stream,
        layer1_weights,
        layer1_bias,
        layer2_weights,
        layer2_bias
    );
    
    // Check outputs
    std::cout << "Neural Network Inference Results:" << std::endl;
    for (int i = 0; i < MAX_OUTPUT_SIZE; i++) {
        if (output_stream.empty()) {
            std::cout << "Error: Output stream empty before reading all expected outputs" << std::endl;
            return 1;
        }
        
        axi_data_t output_data = output_stream.read();
        std::cout << "Output " << i << ": " << output_data.data << std::endl;
    }
    
    if (!output_stream.empty()) {
        std::cout << "Error: Output stream contains more data than expected" << std::endl;
        return 1;
    }
    
    std::cout << "Testbench completed successfully!" << std::endl;
    return 0;
}
