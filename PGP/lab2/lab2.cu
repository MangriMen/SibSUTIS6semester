#include <iostream>
#include <string>
#include <chrono>
#include <fstream>

using namespace std;

#define CUDA_CHECK_RETURN(value)                                          \
    {                                                                     \
        cudaError_t _m_cudaStat = value;                                  \
        if (_m_cudaStat != cudaSuccess)                                   \
        {                                                                 \
            fprintf(stderr, "Error %s at line %d in file %s\n",           \
                    cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__); \
            exit(1);                                                      \
        }                                                                 \
    }

__global__ void vector_add(float a[], float b[])
{
    a[threadIdx.x + blockDim.x * blockIdx.x] += b[threadIdx.x + blockDim.x * blockIdx.x];
}

int main(int argc, char *argv[])
{
    ofstream file_out("out.csv");
    cout.rdbuf(file_out.rdbuf());

    size_t start_threads = 2;
    size_t max_threads = 1 << 10;

    if (argc > 2)
    {
        start_threads = stoll(argv[1]);
        max_threads = stoll(argv[1]);
    }

    size_t N = pow(2, 20);
    for (size_t threads = start_threads; threads <= max_threads; threads = threads << 1)
    {
        size_t blocks = N / threads;

        float *arr1 = new float[N];
        float *arr2 = new float[N];
        float *result = new float[N];

        float *cu_arr1 = nullptr;
        float *cu_arr2 = nullptr;

        for (size_t i = 0; i < N; i++)
        {
            arr1[i] = static_cast<float>(i);
            arr2[i] = static_cast<float>(i);
        }

        CUDA_CHECK_RETURN(cudaMalloc((void **)&cu_arr1, N * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&cu_arr2, N * sizeof(float)))
        CUDA_CHECK_RETURN(cudaMemcpy(cu_arr1, arr1, N * sizeof(float), cudaMemcpyHostToDevice))
        CUDA_CHECK_RETURN(cudaMemcpy(cu_arr2, arr2, N * sizeof(float), cudaMemcpyHostToDevice))

        cudaEvent_t start;
        cudaEvent_t stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        vector_add<<<dim3(blocks), dim3(threads)>>>(cu_arr1, cu_arr2);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_time = 0;
        cudaEventElapsedTime(&elapsed_time, start, stop);

        cout
            << threads
            << ";"
            << elapsed_time
            << endl;

        CUDA_CHECK_RETURN(cudaMemcpy(result, cu_arr1, N * sizeof(float), cudaMemcpyDeviceToHost))

        for (size_t i = 0; i < N; i++)
        {
            if (result[i] != arr1[i] + arr2[i])
            {
                cout << "result[i] != arr1[i] + arr2[i]" << endl;
                return -1;
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(&cu_arr1);
        cudaFree(&cu_arr2);

        delete[] arr1;
        delete[] arr2;
        delete[] result;
    }

    return 0;
}