#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

struct saxpy_functor
{
    const float a;
    saxpy_functor(float _a) : a(_a) {}
    __host__ __device__ float operator()(float x, float y)
    {
        return a * x + y;
    }
};

void saxpy(float a, thrust::device_vector<float> &x,
           thrust::device_vector<float> &y)
{
    saxpy_functor func(a);
    thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}

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

    thrust::host_vector<float> h1(1 << 24);
    thrust::host_vector<float> h2(1 << 24);
    thrust::sequence(h1.begin(), h1.end());
    thrust::fill(h2.begin(), h2.end(), 0);
    thrust::device_vector<float> d1 = h1;
    thrust::device_vector<float> d2 = h2;

    cudaEventRecord(startFunc);

    saxpy(2.0F, d1, d2);

    cudaEventRecord(endFunc);

    h2 = d2;
    h1 = d1;

    cudaEventRecord(endAll);

    for (int i = 0; i < (1 << 8); i++)
    {
        printf("%g\t%g\n", h1[i], h2[i]);
    }

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

    return 0;
}