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

int main(void)
{
    cudaDeviceProp prop;
    int whichDevice;
    CUDA_CHECK_RETURN(cudaGetDevice(&whichDevice));
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, whichDevice));
    if (!prop.deviceOverlap)
    {
        printf("Device will not handle overlaps, so no "
               "speed up from streams\n");
        return 0;
    }

    cudaEvent_t start, stop;
    float elapsedTime;
    // start the timers
    CUDA_CHECK_RETURN(cudaEventCreate(&start));
    CUDA_CHECK_RETURN(cudaEventCreate(&stop));
    CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

    // initialize the stream
    cudaStream_t stream;
    CUDA_CHECK_RETURN(cudaStreamCreate(&stream));

    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;
    // allocate the memory on the GPU
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_a,
                                 N * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_b,
                                 N * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&dev_c,
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
    for (int i = 0; i < FULL_DATA_SIZE; i += N)
    {
        // copy the locked memory to the device, async
        CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_a, host_a + i,
                                          N * sizeof(int),
                                          cudaMemcpyHostToDevice,
                                          stream));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_b, host_b + i,
                                          N * sizeof(int),
                                          cudaMemcpyHostToDevice,
                                          stream));
        kernel<<<N / 256, 256, 0, stream>>>(dev_a, dev_b, dev_c);
        // copy the data from device to locked memory
        CUDA_CHECK_RETURN(cudaMemcpyAsync(host_c + i, dev_c,
                                          N * sizeof(int),
                                          cudaMemcpyDeviceToHost,
                                          stream));
    }

    // copy result chunk from locked to full buffer
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));

    CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime,
                                           start, stop));
    printf("Time taken: %3.1f ms\n", elapsedTime);
    // cleanup the streams and memory
    CUDA_CHECK_RETURN(cudaFreeHost(host_a));
    CUDA_CHECK_RETURN(cudaFreeHost(host_b));
    CUDA_CHECK_RETURN(cudaFreeHost(host_c));
    CUDA_CHECK_RETURN(cudaFree(dev_a));
    CUDA_CHECK_RETURN(cudaFree(dev_b));
    CUDA_CHECK_RETURN(cudaFree(dev_c));

    CUDA_CHECK_RETURN(cudaStreamDestroy(stream));
    return 0;
}