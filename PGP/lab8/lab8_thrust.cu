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
    thrust::host_vector<float> h1(1 << 24);
    thrust::host_vector<float> h2(1 << 24);
    thrust::sequence(h1.begin(), h1.end());
    thrust::fill(h2.begin(), h2.end(), 0);
    thrust::device_vector<float> d1 = h1;
    thrust::device_vector<float> d2 = h2;

    saxpy(2.0F, d1, d2);

    h2 = d2;
    h1 = d1;

    for (int i = 0; i < (1 << 8); i++)
    {
        printf("%g\t%g\n", h1[i], h2[i]);
    }

    return 0;
}