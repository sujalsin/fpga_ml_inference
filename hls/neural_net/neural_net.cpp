#include "neural_net.hpp"

void dense_layer(
    hls::stream<axi_data_t>& input_stream,
    hls::stream<axi_data_t>& output_stream,
    const fixed_t weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE],
    const fixed_t bias[MAX_LAYER_SIZE],
    const int input_size,
    const int output_size
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE bram port=weights
    #pragma HLS INTERFACE bram port=bias
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=16 dim=2
    
    fixed_t input_buffer[MAX_LAYER_SIZE];
    fixed_t output_buffer[MAX_LAYER_SIZE];
    
    // Read input data
    for (int i = 0; i < input_size; i++) {
        #pragma HLS PIPELINE
        axi_data_t input_data = input_stream.read();
        input_buffer[i] = fixed_t(input_data.data);
    }
    
    // Matrix multiplication and bias addition
    for (int i = 0; i < output_size; i++) {
        #pragma HLS PIPELINE
        fixed_t sum = bias[i];
        for (int j = 0; j < input_size; j++) {
            #pragma HLS UNROLL factor=16
            sum += weights[i][j] * input_buffer[j];
        }
        output_buffer[i] = sum;
    }
    
    // Write output data
    for (int i = 0; i < output_size; i++) {
        #pragma HLS PIPELINE
        axi_data_t output_data;
        output_data.data = output_buffer[i].to_int();
        output_data.keep = 1;
        output_data.strb = 1;
        output_data.last = (i == output_size-1);
        output_stream.write(output_data);
    }
}

void relu_layer(
    hls::stream<axi_data_t>& input_stream,
    hls::stream<axi_data_t>& output_stream,
    const int size
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE
        axi_data_t input_data = input_stream.read();
        axi_data_t output_data;
        
        fixed_t val = fixed_t(input_data.data);
        output_data.data = (val > 0) ? val.to_int() : 0;
        output_data.keep = input_data.keep;
        output_data.strb = input_data.strb;
        output_data.last = input_data.last;
        
        output_stream.write(output_data);
    }
}

void batch_norm_layer(
    hls::stream<axi_data_t>& input_stream,
    hls::stream<axi_data_t>& output_stream,
    const fixed_t gamma[MAX_LAYER_SIZE],
    const fixed_t beta[MAX_LAYER_SIZE],
    const fixed_t moving_mean[MAX_LAYER_SIZE],
    const fixed_t moving_var[MAX_LAYER_SIZE],
    const int size
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE bram port=gamma
    #pragma HLS INTERFACE bram port=beta
    #pragma HLS INTERFACE bram port=moving_mean
    #pragma HLS INTERFACE bram port=moving_var
    
    const fixed_t epsilon = 0.001;
    
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE
        axi_data_t input_data = input_stream.read();
        axi_data_t output_data;
        
        fixed_t val = fixed_t(input_data.data);
        fixed_t normalized = (val - moving_mean[i]) / hls::sqrt(moving_var[i] + epsilon);
        fixed_t scaled = gamma[i] * normalized + beta[i];
        
        output_data.data = scaled.to_int();
        output_data.keep = input_data.keep;
        output_data.strb = input_data.strb;
        output_data.last = input_data.last;
        
        output_stream.write(output_data);
    }
}

void neural_net_inference(
    hls::stream<axi_data_t>& input_stream,
    hls::stream<axi_data_t>& output_stream,
    const fixed_t layer1_weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE],
    const fixed_t layer1_bias[MAX_LAYER_SIZE],
    const fixed_t layer2_weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE],
    const fixed_t layer2_bias[MAX_LAYER_SIZE]
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE bram port=layer1_weights
    #pragma HLS INTERFACE bram port=layer1_bias
    #pragma HLS INTERFACE bram port=layer2_weights
    #pragma HLS INTERFACE bram port=layer2_bias
    
    #pragma HLS DATAFLOW
    
    hls::stream<axi_data_t> layer1_output("layer1_output");
    hls::stream<axi_data_t> relu1_output("relu1_output");
    
    // Layer 1: Dense + ReLU
    dense_layer(input_stream, layer1_output, layer1_weights, layer1_bias, MAX_INPUT_SIZE, MAX_LAYER_SIZE);
    relu_layer(layer1_output, relu1_output, MAX_LAYER_SIZE);
    
    // Layer 2: Dense (Output Layer)
    dense_layer(relu1_output, output_stream, layer2_weights, layer2_bias, MAX_LAYER_SIZE, MAX_OUTPUT_SIZE);
}
