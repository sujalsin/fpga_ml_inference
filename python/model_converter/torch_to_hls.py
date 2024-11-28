import torch
import numpy as np
import argparse
import json

def quantize_weights(weights, bits=16, int_bits=8):
    """Quantize floating point weights to fixed point."""
    scale = 2 ** (bits - int_bits - 1)
    weights_scaled = np.clip(weights * scale, -(2**(bits-1)), 2**(bits-1)-1)
    return np.round(weights_scaled).astype(np.int16)

def convert_linear_layer(layer):
    """Convert PyTorch linear layer to HLS format."""
    weights = layer.weight.detach().numpy()
    bias = layer.bias.detach().numpy()
    
    # Quantize weights and bias
    weights_fixed = quantize_weights(weights)
    bias_fixed = quantize_weights(bias)
    
    return {
        'weights': weights_fixed.tolist(),
        'bias': bias_fixed.tolist(),
        'input_size': weights.shape[1],
        'output_size': weights.shape[0]
    }

def convert_model_to_hls(model_path, output_path):
    """Convert PyTorch model to HLS format."""
    # Load PyTorch model
    model = torch.load(model_path)
    model.eval()
    
    # Extract layers
    layers = []
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            layer_data = convert_linear_layer(module)
            layer_data['type'] = 'linear'
            layers.append(layer_data)
        elif isinstance(module, torch.nn.ReLU):
            layers.append({'type': 'relu'})
    
    # Save to JSON format
    with open(output_path, 'w') as f:
        json.dump({
            'layers': layers,
            'input_size': layers[0]['input_size'],
            'output_size': layers[-1]['output_size']
        }, f, indent=2)

def generate_hls_header(json_path, output_path):
    """Generate HLS header file with weights and biases."""
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    # Generate C++ header content
    header_content = """#ifndef MODEL_WEIGHTS_HPP
#define MODEL_WEIGHTS_HPP

#include "neural_net.hpp"

// Model architecture
const int MODEL_INPUT_SIZE = {input_size};
const int MODEL_OUTPUT_SIZE = {output_size};

""".format(**model_data)
    
    # Add weights and biases for each layer
    for i, layer in enumerate(model_data['layers']):
        if layer['type'] == 'linear':
            header_content += f"// Layer {i} weights\n"
            header_content += f"const fixed_t LAYER_{i}_WEIGHTS[MAX_LAYER_SIZE][MAX_LAYER_SIZE] = {{\n"
            for row in layer['weights']:
                header_content += "    {" + ", ".join(str(w) for w in row) + "},\n"
            header_content += "};\n\n"
            
            header_content += f"// Layer {i} bias\n"
            header_content += f"const fixed_t LAYER_{i}_BIAS[MAX_LAYER_SIZE] = {{\n"
            header_content += "    " + ", ".join(str(b) for b in layer['bias']) + "\n"
            header_content += "};\n\n"
    
    header_content += "#endif // MODEL_WEIGHTS_HPP\n"
    
    # Write header file
    with open(output_path, 'w') as f:
        f.write(header_content)

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to HLS format')
    parser.add_argument('model_path', help='Path to PyTorch model file')
    parser.add_argument('output_dir', help='Output directory for HLS files')
    args = parser.parse_args()
    
    # Convert model to JSON
    json_path = f"{args.output_dir}/model_data.json"
    convert_model_to_hls(args.model_path, json_path)
    
    # Generate HLS header
    header_path = f"{args.output_dir}/model_weights.hpp"
    generate_hls_header(json_path, header_path)
    
    print(f"Model converted successfully!")
    print(f"JSON data saved to: {json_path}")
    print(f"HLS header saved to: {header_path}")

if __name__ == "__main__":
    main()
