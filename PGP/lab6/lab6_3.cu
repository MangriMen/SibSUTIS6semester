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

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)
__global__ void kernel(int *a, int *b, int *c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main()
{
    cudaDeviceProp prop;
    int whichDevice;
    CUDA_CHECK_RETURN(cudaGetDevice(&whichDevice));
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, whichDevice));

    if (!prop.deviceOverlap)
    {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }
    cudaEvent_t start, stop;
    float elapsedTime;
    // start the timers
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

    // initialize the streams
    cudaStream_t stream0, stream1;
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream0));
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));

    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0; // GPU buffers for stream0
    int *dev_a1, *dev_b1, *dev_c1; // GPU buffers for stream1
    // allocate the memory on the GPU
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_a0,
                                 N * sizeof(int)));

    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_b0,
                                 N * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_c0,
                                 N * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_a1,
                                 N * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_b1,
                                 N * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_c1,
                                 N * sizeof(int)));
    // allocate page-locked memory, used to stream
    CUDA_CHECK_RETURN(cudaHostAlloc((void **)&host_a,
                                    FULL_DATA_SIZE * sizeof(int),
                                    cudaHostAllocDefault));
    CUDA_CHECK_RETURN(cudaHostAlloc((void **)&host_b,
                                    FULL_DATA_SIZE * sizeof(int),
                                    cudaHostAllocDefault));
    CUDA_CHECK_RETURN(cudaHostAlloc((void **)&host_c,
                                    FULL_DATA_SIZE * sizeof(int),
                                    cudaHostAllocDefault));
    for (int i = 0; i < FULL_DATA_SIZE; i++)
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    // now loop over full data, in bite-sized chunks
    for (int i = 0; i < FULL_DATA_SIZE; i += N * 2)
    {
        // enqueue copies of a in stream0 and stream1
        CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a0, host_a + i,
                                          N * sizeof(int),
                                          cudaMemcpyHostToDevice,
                                          stream0));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a1, host_a + i + N,
                                          N * sizeof(int),
                                          cudaMemcpyHostToDevice,
                                          stream1));
        // enqueue copies of b in stream0 and stream1
        CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b0, host_b + i,
                                          N * sizeof(int),
                                          cudaMemcpyHostToDevice,
                                          stream0));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b1, host_b + i + N,
                                          N * sizeof(int),
                                          cudaMemcpyHostToDevice,
                                          stream1));
        // enqueue kernels in stream0 and stream1
        kernel<<<N / 256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
        kernel<<<N / 256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
        // enqueue copies of c from device to locked memory
        CUDA_CHECK_RETURN(cudaMemcpyAsync(host_c + i, dev_c0,
                                          N * sizeof(int),
                                          cudaMemcpyDeviceToHost,
                                          stream0));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(host_c + i + N, dev_c1,
                                          N * sizeof(int),
                                          cudaMemcpyDeviceToHost,
                                          stream1));
    }

    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream0));
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));

    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime,
                                           start, stop));
    printf("Time taken: %3.1f ms\n", elapsedTime);
    // cleanup the streams and memory
    CUDA_CHECK_RETURN(cudaFreeHost(host_a));
    CUDA_CHECK_RETURN(cudaFreeHost(host_b));
    CUDA_CHECK_RETURN(cudaFreeHost(host_c));

    CUDA_CHECK_RETURN(cudaFree(dev_a0));
    CUDA_CHECK_RETURN(cudaFree(dev_b0));
    CUDA_CHECK_RETURN(cudaFree(dev_c0));
    CUDA_CHECK_RETURN(cudaFree(dev_a1));
    CUDA_CHECK_RETURN(cudaFree(dev_b1));
    CUDA_CHECK_RETURN(cudaFree(dev_c1));
    CUDA_CHECK_RETURN(cudaStreamDestroy(stream0));
    CUDA_CHECK_RETURN(cudaStreamDestroy(stream1));

    return 0;
}