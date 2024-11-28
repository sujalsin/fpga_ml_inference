#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// Fixed-point data type definitions
typedef ap_fixed<16,8> fixed_t;  // 16-bit fixed-point, 8 integer bits
typedef ap_axiu<16,1,1,1> axi_data_t;

// Maximum layer dimensions
const int MAX_LAYER_SIZE = 256;
const int MAX_INPUT_SIZE = 64;
const int MAX_OUTPUT_SIZE = 32;

// Neural network layer interfaces
void dense_layer(
    hls::stream<axi_data_t>& input_stream,
    hls::stream<axi_data_t>& output_stream,
    const fixed_t weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE],
    const fixed_t bias[MAX_LAYER_SIZE],
    const int input_size,
    const int output_size
);

void relu_layer(
    hls::stream<axi_data_t>& input_stream,
    hls::stream<axi_data_t>& output_stream,
    const int size
);

void batch_norm_layer(
    hls::stream<axi_data_t>& input_stream,
    hls::stream<axi_data_t>& output_stream,
    const fixed_t gamma[MAX_LAYER_SIZE],
    const fixed_t beta[MAX_LAYER_SIZE],
    const fixed_t moving_mean[MAX_LAYER_SIZE],
    const fixed_t moving_var[MAX_LAYER_SIZE],
    const int size
);

// Top-level inference function
void neural_net_inference(
    hls::stream<axi_data_t>& input_stream,
    hls::stream<axi_data_t>& output_stream,
    const fixed_t layer1_weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE],
    const fixed_t layer1_bias[MAX_LAYER_SIZE],
    const fixed_t layer2_weights[MAX_LAYER_SIZE][MAX_LAYER_SIZE],
    const fixed_t layer2_bias[MAX_LAYER_SIZE]
);

#endif // NEURAL_NET_HPP
