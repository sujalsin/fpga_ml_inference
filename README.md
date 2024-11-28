# FPGA-Based Machine Learning Inference Engine

A high-performance neural network inference engine implemented on FPGA, specifically designed for low-latency trading applications. This project demonstrates the implementation of hardware-accelerated machine learning models using High-Level Synthesis (HLS) and custom Verilog modules.

```ascii
                                     FPGA Implementation
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│    ┌─────────────┐    ┌──────────┐    ┌────────────┐    ┌───────┐  │
│    │             │    │          │    │            │    │       │  │
│ ───►  AXI4-Stream├───►│   Dense  ├───►│    ReLU    ├───►│ Dense │──►│
│    │  Interface  │    │  Layer   │    │            │    │       │  │
│    │             │    │          │    │            │    │       │  │
│    └─────────────┘    └──────────┘    └────────────┘    └───────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Features

### Hardware Optimization
- **Pipelined Architecture**: Achieves high throughput with minimal latency
- **Fixed-Point Arithmetic**: Optimized numerical precision for FPGA implementation
- **Parallel Processing**: Multiple processing elements for concurrent computation
- **Memory Management**: Efficient on-chip memory utilization and access patterns

### Supported Layer Types
- Dense (Fully Connected) Layers
- ReLU Activation
- Batch Normalization
- Configurable Layer Sizes

### Interface
- AXI4-Stream for high-speed data transfer
- Memory-mapped configuration interface
- Status and control registers

## Project Structure

```
├── hls/                    # HLS source files
│   ├── neural_net/        # Neural network layer implementations
│   │   ├── neural_net.hpp # Core definitions and interfaces
│   │   └── neural_net.cpp # Layer implementations
│   └── test/              # HLS testbenches
├── rtl/                   # Verilog/SystemVerilog files
│   ├── src/              # RTL source files
│   └── tb/               # RTL testbenches
├── python/               # Python utilities
│   └── model_converter/  # Scripts for converting ML models to HLS
└── docs/                # Documentation
```

## Performance Metrics

```ascii
┌────────────────┬────────────┐
│ Metric         │ Value      │
├────────────────┼────────────┤
│ Clock Speed    │ 200 MHz    │
│ Latency        │ < 500 ns   │
│ Throughput     │ 1M inf/sec │
│ Precision      │ 16-bit     │
│ Resource Usage │ ~30% FPGA  │
└────────────────┴────────────┘
```

## Example Use Cases

### 1. High-Frequency Trading
```ascii
Market Data ──► FPGA Inference ──► Trading Signal
   (1 μs)         (500 ns)          (100 ns)
```
- Predicts short-term price movements
- Ultra-low latency decision making
- Real-time risk assessment

### 2. Market Making
```ascii
┌─────────────────┐
│ Bid/Ask Spread  │
│   Prediction    │
└────────┬────────┘
         │
    FPGA Model
         │
┌────────┴────────┐
│  Quote Updates  │
│    (< 1 μs)     │
└─────────────────┘
```
- Optimal spread determination
- Inventory management
- Risk-adjusted pricing

### 3. Portfolio Optimization
```ascii
Input Features    Model Layers      Output
   [64x1]           [256]          [32x1]
     │                │              │
     ▼                ▼              ▼
┌──────────┐    ┌──────────┐    ┌───────┐
│ Market   │───►│ Hidden   │───►│ Asset │
│ Features │    │ Layers   │    │Weights│
└──────────┘    └──────────┘    └───────┘
```
- Real-time portfolio rebalancing
- Risk factor analysis
- Multi-asset optimization

## Requirements

- Vivado HLS 2021.2 or later
- Python 3.8+
- PyTorch (for model conversion)
- Vivado Design Suite
- C++ compiler supporting C++14 or later

## Getting Started

1. **Clone the Repository**
```bash
git clone <repository-url>
cd fpga_ml_inference
```

2. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```

3. **Convert PyTorch Model**
```bash
cd python/model_converter
python torch_to_hls.py <model_path> ../../hls/neural_net
```

4. **Build and Test**
```bash
cd ../../hls
make test
make synthesis
make export
```

## Performance Optimization Tips

1. **Data Quantization**
```cpp
// Example fixed-point configuration
typedef ap_fixed<16,8> fixed_t;  // 16-bit total, 8 integer bits
```

2. **Pipeline Configuration**
```cpp
#pragma HLS PIPELINE II=1  // Initialize new input every clock cycle
```

3. **Memory Partitioning**
```cpp
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=16
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

- Xilinx HLS Documentation
- PyTorch Team
- FPGA ML Research Community
