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
    const int n = 1 << 24;

    cudaEvent_t startGlobal, endGlobal, startLocal, endLocal;
    cudaEventCreate(&startGlobal);
    cudaEventCreate(&endGlobal);
    cudaEventCreate(&startLocal);
    cudaEventCreate(&endLocal);

    cudaEventRecord(startGlobal);

    float *hostArr = (float *)malloc(n * sizeof(float));
    float *hostArrRes = (float *)malloc(n * sizeof(float));

    float *cudaArr = NULL;
    cudaMalloc((void **)&cudaArr, (n) * sizeof(float));

    float *cudaArrRes = NULL;
    cudaMalloc((void **)&cudaArrRes, (n) * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        hostArr[i] = i;
    }

    cudaMemcpy(cudaArr, hostArr, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(startLocal);
    saxpy<<<4096, 256>>>(n, 2.0, cudaArr, cudaArrRes);
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

    cudaEventDestroy(startGlobal);
    cudaEventDestroy(endGlobal);
    cudaEventDestroy(startLocal);
    cudaEventDestroy(endLocal);

    free(hostArr);
    free(hostArrRes);
    free(cudaArr);
    free(cudaArrRes);

    return EXIT_SUCCESS;
}