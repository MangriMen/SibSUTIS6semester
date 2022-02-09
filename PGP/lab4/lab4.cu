#include <iostream>
#include <string>

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

__global__ void gInitializeStorage(float *storage_d)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int N = blockDim.x * gridDim.x;

    storage_d[i + j * N] = (float)(i + j * N);
}

__global__ void gInitializeStorageInvert(float *storage_d)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int N = blockDim.x * gridDim.x;

    storage_d[j + i * N] = (float)(j + i * N);
}

__global__ void gTranspose0(float *storage_d, float *storage_d_t)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int N = blockDim.x * gridDim.x;

    storage_d_t[j + i * N] = storage_d[i + j * N];
}

__global__ void gTranspose11(float *storage_d, float *storage_d_t)
{
    extern __shared__ float buffer[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int N = blockDim.x * gridDim.x;

    buffer[threadIdx.y + threadIdx.x * blockDim.y] = storage_d[i + j * N];

    __syncthreads();

    i = threadIdx.x + blockIdx.y * blockDim.x;
    j = threadIdx.y + blockIdx.x * blockDim.y;

    storage_d_t[i + j * N] = buffer[threadIdx.x + threadIdx.y * blockDim.x];
}

#define SH_DIM 32
__global__ void gTranspose12(float *storage_d, float *storage_d_t)
{
    __shared__ float buffer_s[SH_DIM][SH_DIM];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int N = blockDim.x * gridDim.x;
    buffer_s[threadIdx.y][threadIdx.x] = storage_d[i + j * N];
    __syncthreads();
    i = threadIdx.x + blockIdx.y * blockDim.x;
    j = threadIdx.y + blockIdx.x * blockDim.y;
    storage_d_t[i + j * N] = buffer_s[threadIdx.x][threadIdx.y];
}

__global__ void gTranspose2(float *storage_d, float *storage_d_t)
{
    __shared__ float buffer[SH_DIM][SH_DIM + 1];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int N = blockDim.x * gridDim.x;

    buffer[threadIdx.y][threadIdx.x] = storage_d[i + j * N];

    __syncthreads();

    i = threadIdx.x + blockIdx.y * blockDim.x;
    j = threadIdx.y + blockIdx.x * blockDim.y;
    storage_d_t[i + j * N] = buffer[threadIdx.x][threadIdx.y];
}

void matrixPrint(float *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            cout << matrix[j + i * size] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char *argv[])
{
    // Init
    if (argc < 3)
    {
        cerr << "USAGE: matrix <dimension of matrix> <dimension of threads>\n";
        return -1;
    }

    size_t N = atoi(argv[1]);
    size_t dim_of_threads = atoi(argv[2]);

    bool is_out = N <= 32;

    if (N % dim_of_threads)
    {
        cerr << "Change threads count to divide N without remainder";
        return -1;
    }

    size_t dim_of_blocks = N / dim_of_threads;

    const int max_size = 1 << 8;
    if (dim_of_blocks > max_size)
    {
        cerr << "Max blocks count is 256";
        return -1;
    }

    // Memory allocation
    float *storage_d = nullptr;
    float *storage_d_t = nullptr;
    float *storage_h = nullptr;

    CUDA_CHECK_RETURN(cudaMalloc((void **)&storage_d, N * N * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&storage_d_t, N * N * sizeof(float)));
    storage_h = new float[N * N];

    float elapsed_time_initialize = 0;
    float elapsed_time_initialize_invert = 0;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEvent_t startInvert;
    cudaEvent_t stopInvert;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventCreate(&startInvert);
    cudaEventCreate(&stopInvert);

    cudaEventRecord(start, 0);
    gInitializeStorage<<<dim3(dim_of_blocks, dim_of_blocks),
                         dim3(dim_of_threads, dim_of_threads)>>>(storage_d);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time_initialize, start, stop);

    cudaEventRecord(startInvert, 0);
    gInitializeStorageInvert<<<dim3(dim_of_blocks, dim_of_blocks),
                               dim3(dim_of_threads, dim_of_threads)>>>(storage_d);
    cudaDeviceSynchronize();
    cudaEventRecord(stopInvert, 0);
    cudaEventSynchronize(stopInvert);

    cudaEventElapsedTime(&elapsed_time_initialize_invert, startInvert, stopInvert);

    if (is_out)
    {
        cout << "Default;" << elapsed_time_initialize << endl;
        cout << "Invert;" << elapsed_time_initialize_invert << endl;
        cout << endl;
    }

    fill(storage_h, storage_h + (N * N), 0.0f);
    CUDA_CHECK_RETURN(cudaMemcpy(storage_h, storage_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (is_out)
    {
        cout << "Default matrix" << endl;
        matrixPrint(storage_h, N);
    }

    // gTranspose0
    gTranspose0<<<dim3(dim_of_blocks, dim_of_blocks),
                  dim3(dim_of_threads, dim_of_threads)>>>(storage_d, storage_d_t);
    cudaDeviceSynchronize();

    fill(storage_h, storage_h + (N * N), 0.0f);
    CUDA_CHECK_RETURN(cudaMemcpy(storage_h, storage_d_t, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (is_out)
    {
        cout << "gTranspose0" << endl;
        matrixPrint(storage_h, N);
    }

    // gTranspose 1
    gTranspose11<<<dim3(dim_of_blocks, dim_of_blocks),
                   dim3(dim_of_threads, dim_of_threads),
                   dim_of_threads * dim_of_threads * sizeof(float)>>>(storage_d, storage_d_t);
    cudaDeviceSynchronize();

    fill(storage_h, storage_h + (N * N), 0.0f);
    CUDA_CHECK_RETURN(cudaMemcpy(storage_h, storage_d_t, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (is_out)
    {
        cout << "gTranspose1" << endl;
        matrixPrint(storage_h, N);
    }

    // gTranspose12
    gTranspose12<<<dim3(dim_of_blocks, dim_of_blocks),
                   dim3(dim_of_threads, dim_of_threads)>>>(storage_d, storage_d_t);
    cudaDeviceSynchronize();

    fill(storage_h, storage_h + (N * N), 0.0f);
    cudaMemcpy(storage_h, storage_d_t, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (is_out)
    {
        cout << "gTranspose12" << endl;
        matrixPrint(storage_h, N);
    }

    // gTranspose2
    gTranspose2<<<dim3(dim_of_blocks, dim_of_blocks),
                  dim3(dim_of_threads, dim_of_threads)>>>(storage_d, storage_d_t);
    cudaDeviceSynchronize();

    fill(storage_h, storage_h + (N * N), 0.0f);
    cudaMemcpy(storage_h, storage_d_t, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (is_out)
    {
        cout << "gTranspose2" << endl;
        matrixPrint(storage_h, N);
    }

    // Clearing memory
    cudaFree(storage_d);
    cudaFree(storage_d_t);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startInvert);
    cudaEventDestroy(stopInvert);

    delete[] storage_h;

    return 0;
}