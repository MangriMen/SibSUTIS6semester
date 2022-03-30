#include <iostream>

__global__ void saxpy(int n, float a, float *__restrict x, float *__restrict y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        y[i] = a * x[i] + y[i];
    }
}

int main()
{
    const int n = 1 << 20;

    cudaEvent_t startGlobal, stopGlobal, startLocal, stopLocal;
    cudaEventCreate(&startGlobal);
    cudaEventCreate(&stopGlobal);
    cudaEventCreate(&startLocal);
    cudaEventCreate(&stopLocal);

    cudaEventRecord(startGlobal);

    float *hostArr = (float *)malloc(n * sizeof(float));
    float *hostArrRes = (float *)malloc(n * sizeof(float));

    float *cudaArr = NULL;
    float *cudaArrRes = NULL;
    cudaMalloc((void **)&cudaArr, (n) * sizeof(float));
    cudaMalloc((void **)&cudaArrRes, (n) * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        hostArr[i] = i;
    }

    cudaMemcpy(cudaArr, hostArr, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(startLocal);
    saxpy<<<4096, 256>>>(n, 2.0, cudaArr, cudaArrRes);
    cudaEventRecord(stopLocal);
    cudaEventSynchronize(stopLocal);

    cudaMemcpy(hostArrRes, cudaArrRes + 1, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stopGlobal);
    cudaEventSynchronize(stopGlobal);

    float globalElapsedTime = 0;
    cudaEventElapsedTime(&globalElapsedTime, startGlobal, stopGlobal);

    float localElapsedTime = 0;
    cudaEventElapsedTime(&localElapsedTime, startLocal, stopLocal);

    printf("Native\n\tGlobal: %f ms\n\tLocal: %f ms\n", globalElapsedTime, localElapsedTime);

    cudaEventDestroy(startGlobal);
    cudaEventDestroy(stopGlobal);
    cudaEventDestroy(startLocal);
    cudaEventDestroy(stopLocal);

    free(hostArr);
    free(hostArrRes);
    free(cudaArr);
    free(cudaArrRes);

    return EXIT_SUCCESS;
}