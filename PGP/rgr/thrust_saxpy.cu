#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <fstream>

using namespace std;

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

int main(int argc, char *argv[])
{
    int n = 1 << 24;

    if (argc > 1)
    {
        n = stoll(argv[1]);
    }
    else
    {
        return EXIT_FAILURE;
    }

    cudaEvent_t startGlobal;
    cudaEvent_t endGlobal;
    cudaEvent_t startLocal;
    cudaEvent_t endLocal;

    cudaEventCreate(&startGlobal);
    cudaEventCreate(&endGlobal);
    cudaEventCreate(&startLocal);
    cudaEventCreate(&endLocal);

    cudaEventRecord(startGlobal);

    thrust::host_vector<float> h1(n);
    thrust::host_vector<float> h2(n);
    thrust::sequence(h1.begin(), h1.end());
    thrust::fill(h2.begin(), h2.end(), 0);
    thrust::device_vector<float> d1 = h1;
    thrust::device_vector<float> d2 = h2;

    cudaEventRecord(startLocal);
    saxpy(2.0F, d1, d2);
    cudaEventRecord(endLocal);
    cudaEventSynchronize(endLocal);

    h2 = d2;
    h1 = d1;

    cudaEventRecord(endGlobal);

    float globalElapsedTime = 0;
    cudaEventElapsedTime(&globalElapsedTime, startGlobal, endGlobal);

    float localElapsedTime = 0;
    cudaEventElapsedTime(&localElapsedTime, startLocal, endLocal);

    printf("Thrust\n\tGlobal: %f ms\n\tLocal: %f ms\n", globalElapsedTime, localElapsedTime);

    ofstream fileOut("result.csv", ios::app);
    fileOut << "thrust;" << globalElapsedTime << ";" << localElapsedTime << "\n";
    fileOut.close();

    cudaEventDestroy(startGlobal);
    cudaEventDestroy(endGlobal);
    cudaEventDestroy(startLocal);
    cudaEventDestroy(endLocal);

    return EXIT_SUCCESS;
}