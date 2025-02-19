#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

char* readFile(const char* file_path, long* file_size) {
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* file_data = (char*)malloc(*file_size + 1);
    if (file_data == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }
    fread(file_data, 1, *file_size, file);
    file_data[*file_size] = '\0';

    fclose(file);

    return file_data;
}

__global__ void xor (char* file_data_xored, char* file_data) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    file_data_xored[x] = file_data[x] ^ 0x43;
}

cudaError_t xor_with_cuda(char* file_data_xored, char* file_data, long file_size) {
    cudaError_t cudaStatus;
    char* dev_file_data;
    char* dev_file_data_xored;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_file_data, file_size * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_file_data_xored, file_size * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_file_data, file_data, file_size * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    int blockSize = 256;
    int numBlocks = (file_size + blockSize - 1) / blockSize;
    xor << <numBlocks, blockSize >> > (dev_file_data_xored, dev_file_data);

    cudaStatus = cudaMemcpy(file_data_xored, dev_file_data_xored, file_size * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

Error:
    printf("test");
    cudaFree(dev_file_data);
    cudaFree(dev_file_data_xored);

    return cudaStatus;
}

void xor_with_cpu(char* file_data, long file_size) {
    if (file_data != NULL) {
        for (int x = 0; x < file_size; x++) {
            file_data[x] = file_data[x] ^ 0x43;
        }
    }
}

int main() {
    const char* file_path = "test_file.txt"; // Change THIS | GPU > CPU : File > 40 Mo
    long file_size;
    char* file_data = readFile(file_path, &file_size);

    char* file_data_xored = (char*)malloc(file_size + 1);

    //  CPU
    auto start_time_cpu = std::chrono::high_resolution_clock::now();

    xor_with_cpu(file_data, file_size); ///

    auto end_time_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_cpu = end_time_cpu - start_time_cpu;
    printf("Temps d'exécution CPU : %f millisecondes\n", elapsed_time_cpu.count());
    printf("file_data:\n%s\n", file_data);
    //  GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    xor_with_cuda(file_data_xored, file_data, file_size); ///

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_gpu;
    cudaEventElapsedTime(&elapsed_time_gpu, start, stop);
    printf("Temps d'exécution GPU : %f millisecondes\n", elapsed_time_gpu);
    printf("file_data:\n%s\n", file_data_xored);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(file_data);
    free(file_data_xored);

    return 0;
}