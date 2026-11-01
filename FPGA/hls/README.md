# HLS Model Implementation

This folder contains the C/C++ implementation of the CNN-LSTM model for High-Level Synthesis (HLS), along with testbenches, reference data, and build configurations.

HLS converts the C/C++ implementation into hardware IP that can be integrated into a Vivado FPGA design.

## Contents

- **conv1 and conv2:** Convolution layers for extracting features from the input signals.
- **relu_max1 and relu_max2:** ReLU activation, which replaces negative values with zero and reduce length with max pooling.
- **lstm:** Processes sequential features to capture relationships over time.
- **linear:** Fully connected layer that maps features to model outputs.
- **Inference code and headers:** Define the model interfaces and connect its processing stages.
- **Golden vectors:** Reference inputs and expected outputs for checking numerical correctness.
- **Testbenches:** Run the implementations against the golden vectors.
- **HLS configuration files:** Specify synthesis and simulation settings.
- **JSON files:** Store supporting Vitis project and development-tool metadata.

## Verification and Build

1. Open or recreate the relevant HLS component using its configuration files.
2. Update local file paths if necessary.
3. Run C simulation with the corresponding testbench and golden vectors.
4. Review the test results before running HLS synthesis.
5. Review synthesis reports for resource usage and latency, then export the generated IP for Vivado integration.

The Vivado project, constraints, and FPGA overlay files are documented separately in [`../vivado/`](../vivado/).
