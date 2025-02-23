# XOR_CUDA_Unpacking_C

## Description
This project demonstrates the use of CUDA to accelerate (>50 Mb binary) XOR-based unpacking of a shellcode compiled with the program. 
It compares CPU and GPU execution times, highlighting the benefits of parallel processing for large binary. Currently, this is a simple proof of concept. A more advanced version will include better decryption algorithms and additional features for a full packer.

## Features
- Allocate a encrypted shellcode from the executable itself
- Performs XOR encryption/decryption on the CPU
- Performs XOR encryption/decryption on the GPU using CUDA
- Measures and compares execution times between CPU and GPU (Use profiler to better results)

## Compatibility
This program works on both Windows and Linux. However, on Windows, parallelism and overlapping may be affected or disabled due to WDDM. Users may try enabling hardware-accelerated scheduling, but this is not a guaranteed solution. The behavior depends on factors such as the Windows version, driver version, and kernel configuration. Updating the GPU driver may help, but it is only confirmed as a driver issue if the update resolves the problem.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler (GCC, MSVC, or Clang)

## Compilation
To compile the project, follow these steps:

### Windows
```sh
ld.exe -r -b binary shellcode.bin -o shellcode_bin.obj --oformat pe-x86-64
```
With Visual Studio:
- Go to `View > Solution Explorer`
- Right-click the project and select `Properties`
- Navigate to `Linker > Input`
- Edit and add `shellcode_bin.obj` to Additional Dependencies

With NVCC:
```sh
nvcc -o kernel kernel.cu shellcode_bin.obj
```

### Linux
```sh
objcopy --input binary --output elf64-x86-64 --binary-architecture i386:x86-64 shellcode.bin shellcode_bin.o
```
With NVCC:
```sh
nvcc -o kernel kernel.cu shellcode_bin.o
```

### Verify the binary
```sh
strings shellcode_bin.o | grep binary
```

## Usage
Run the executable :

```sh
./kernel
```

## Performance Benchmark
The program outputs execution times for both CPU and GPU implementations, demonstrating the efficiency of GPU acceleration for large shellcode (>50 Mb).

## License
This project is open-source and available under the MIT License.

## Author
R3dy

