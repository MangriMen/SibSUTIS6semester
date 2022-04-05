#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <fstream>

using namespace std;

int main()
{
    cudaEvent_t startGlobal;
    cudaEvent_t endGlobal;
    cudaEvent_t startLocal;
    cudaEvent_t endLocal;

    cudaEventCreate(&startGlobal);
    cudaEventCreate(&endGlobal);
    cudaEventCreate(&startLocal);
    cudaEventCreate(&endLocal);

    cudaEventRecord(startGlobal);

    const int n = 1 << 24;
    const size_t size_in_bytes = (n * sizeof(float));

    float *A_dev = NULL;
    cudaMalloc((void **)&A_dev, size_in_bytes);

    float *B_dev = NULL;
    cudaMalloc((void **)&B_dev, size_in_bytes);

    float *A_h = NULL;
    cudaMallocHost((void **)&A_h, size_in_bytes);

    float *B_h = NULL;
    cudaMallocHost((void **)&B_h, size_in_bytes);

    memset(A_h, 0, size_in_bytes);
    memset(B_h, 0, size_in_bytes);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    for (int i = 0; i < n; i++)
    {
        A_h[i] = (float)i;
    }

    const int num_rows = n;
    const int num_cols = 1;
    const size_t elem_size = sizeof(float);

    //Копирование матрицы с числом строк num_elem и одним столбцом с
    //хоста на устройство
    cublasSetMatrix(num_rows, num_cols, elem_size, A_h,
                    num_rows, A_dev, num_rows);

    //Очищаем массив на устройстве
    cudaMemset(B_dev, 0, size_in_bytes);

    // выполнение SingleAlphaXPlusY
    const int stride = 1;
    float alpha = 2.0F;
    cudaEventRecord(startLocal);

    cublasSaxpy(cublas_handle, n, &alpha, A_dev,
                stride, B_dev, stride);

    cudaEventRecord(endLocal);

    //Копирование матриц с числом строк num_elem и одним столбцом с
    //устройства на хост
    cublasGetMatrix(num_rows, num_cols, elem_size, A_dev,
                    num_rows, A_h, num_rows);
    cublasGetMatrix(num_rows, num_cols, elem_size, B_dev,
                    num_rows, B_h, num_rows);

    // Удостоверяемся, что все асинхронные вызовы выполнены
    const int default_stream = 0;
    cudaStreamSynchronize(default_stream);

    cublasDestroy(cublas_handle);
    cudaFree(A_dev);
    cudaFree(B_dev);

    cudaFreeHost(A_h);
    cudaFreeHost(A_h);
    cudaFreeHost(B_h);

    cudaEventRecord(endGlobal);

    float globalElapsedTime = 0;
    cudaEventElapsedTime(&globalElapsedTime, startGlobal, endGlobal);

    float localElapsedTime = 0;
    cudaEventElapsedTime(&localElapsedTime, startLocal, endLocal);

    printf("CUBLAS\n\tGlobal: %f ms\n\tLocal: %f ms\n", globalElapsedTime, localElapsedTime);

    ofstream fileOut("result.csv", ios::app);
    fileOut << "cublas;" << globalElapsedTime << ";" << localElapsedTime << "\n";
    fileOut.close();

    cudaEventDestroy(startGlobal);
    cudaEventDestroy(endGlobal);
    cudaEventDestroy(startLocal);
    cudaEventDestroy(endLocal);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}