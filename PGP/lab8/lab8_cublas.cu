#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

int main()
{
    cudaEvent_t startAll;
    cudaEvent_t startFunc;
    cudaEvent_t endFunc;
    cudaEvent_t endAll;

    cudaEventCreate(&startAll);
    cudaEventCreate(&startFunc);
    cudaEventCreate(&endFunc);
    cudaEventCreate(&endAll);

    cudaEventRecord(startAll);

    const int num_elem = 1 << 24;
    const size_t size_in_bytes = (num_elem * sizeof(float));

    float *A_dev;
    cudaMalloc((void **)&A_dev, size_in_bytes);

    float *B_dev;
    cudaMalloc((void **)&B_dev, size_in_bytes);

    float *A_h;
    cudaMallocHost((void **)&A_h, size_in_bytes);

    float *B_h;
    cudaMallocHost((void **)&B_h, size_in_bytes);

    memset(A_h, 0, size_in_bytes);
    memset(B_h, 0, size_in_bytes);

    // Инициализация библиотеки CUBLAS
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    for (int i = 0; i < num_elem; i++)
    {
        A_h[i] = (float)i;
    }

    const int num_rows = num_elem;
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
    cudaEventRecord(startFunc);

    cublasSaxpy(cublas_handle, num_elem, &alpha, A_dev,
                stride, B_dev, stride);

    cudaEventRecord(endFunc);

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

    cudaEventRecord(endAll);

    float func_time = 0;
    cudaEventElapsedTime(&func_time, startFunc, endFunc);

    float program_time = 0;
    cudaEventElapsedTime(&program_time, startAll, endAll);

    printf("SAXPY: %f\n", func_time);
    printf("Program: %f\n", program_time);

    cudaEventDestroy(startFunc);
    cudaEventDestroy(startAll);
    cudaEventDestroy(endFunc);
    cudaEventDestroy(endAll);

    cudaDeviceReset();

    return 0;
}