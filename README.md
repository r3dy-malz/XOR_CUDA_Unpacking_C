# XOR_CUDA_Unpacking_C

## Description
This project demonstrates how to use CUDA to accelerate XOR-based unpacking of a file. It compares the execution time of XOR operations on the CPU versus the GPU, showing the performance benefits of parallel processing for large files (greater than 40 KB).
For the moment, the program is just a benchmark between CPU/GPU execution speed.

## Features
- Reads a file into memory
- Performs XOR encryption/decryption on the CPU
- Performs XOR encryption/decryption on the GPU using CUDA
- Measures and compares execution times between CPU and GPU

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler (GCC, MSVC, or Clang)
- (In the future, a release with a .exe file)

## Compilation
To compile the project, use the following command:

```sh
nvcc -o xor_cuda_unpacking main.cu
```

## Usage
Run the executable with a test file:

```sh
./xor_cuda_unpacking
```

Make sure to modify `file_path` in the source code to point to the desired file.

## Performance Benchmark
The program outputs execution times for both CPU and GPU implementations, demonstrating the efficiency of GPU acceleration for large files.

## License
This project is open-source and available under the MIT License.

## Author

R3dy

