#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>

using namespace std;

__global__ void saxpy(int n, float a, float *__restrict x, float *__restrict y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        y[i] = a * x[i] + y[i];
    }
}

int main(int argc, char *argv[])
{
    int n = 1 << 24;

    if (argc > 1)
    {
        n = stoi(argv[1]);
    }
    else
    {
        return EXIT_FAILURE;
    }

    cudaEvent_t startGlobal, endGlobal, startLocal, endLocal;
    cudaEventCreate(&startGlobal);
    cudaEventCreate(&endGlobal);
    cudaEventCreate(&startLocal);
    cudaEventCreate(&endLocal);

    cudaEventRecord(startGlobal);

    float *hostArr = (float *)malloc(n * sizeof(float));
    float *hostArrRes = (float *)malloc(n * sizeof(float));

    float *cudaArr = NULL;
    cudaMalloc((void **)&cudaArr, (n + 1) * sizeof(float));

    float *cudaArrRes = NULL;
    cudaMalloc((void **)&cudaArrRes, (n + 1) * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        hostArr[i] = i;
    }

    cudaMemcpy(cudaArr, hostArr, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(startLocal);
    saxpy<<<512, 32>>>(n, 2.0, cudaArr, cudaArrRes);
    cudaEventRecord(endLocal);
    cudaEventSynchronize(endLocal);

    cudaMemcpy(hostArrRes, cudaArrRes + 1, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(endGlobal);
    cudaEventSynchronize(endGlobal);

    float globalElapsedTime = 0;
    cudaEventElapsedTime(&globalElapsedTime, startGlobal, endGlobal);

    float localElapsedTime = 0;
    cudaEventElapsedTime(&localElapsedTime, startLocal, endLocal);

    printf("Native\n\tGlobal: %f ms\n\tLocal: %f ms\n", globalElapsedTime, localElapsedTime);

    ofstream fileOut("result.csv", ios::app);
    fileOut << "native;" << globalElapsedTime << ";" << localElapsedTime << "\n";
    fileOut.close();

    cudaEventDestroy(startGlobal);
    cudaEventDestroy(endGlobal);
    cudaEventDestroy(startLocal);
    cudaEventDestroy(endLocal);

    cudaFree(cudaArr);
    cudaFree(cudaArrRes);

    free(hostArr);
    free(hostArrRes);

    return EXIT_SUCCESS;
}