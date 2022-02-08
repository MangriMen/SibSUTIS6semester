#include <iostream>
#include <string>
#include <chrono>
#include <fstream>

using namespace std;

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

    for (size_t i = 0; i < N; i++)
    {
      arr1[i] = static_cast<float>(i);
      arr2[i] = static_cast<float>(i);
    }

    float *cu_arr1 = nullptr;
    float *cu_arr2 = nullptr;

    if (cudaMalloc((void **)&cu_arr1, N * sizeof(float)) != cudaSuccess)
    {
      cerr << "Error when memory allocation for cuda";
      return -1;
    }

    if (cudaMalloc((void **)&cu_arr2, N * sizeof(float)) != cudaSuccess)
    {
      cerr << "Error when memory allocation for cuda";
      return -1;
    }

    if (cudaMemcpy(cu_arr1, arr1, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
      cerr << "Error when copy memory from host to device";
      return -1;
    }

    if (cudaMemcpy(cu_arr2, arr2, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
      cerr << "Error when copy memory from host to device";
      return -1;
    }

    auto timer_start = chrono::high_resolution_clock::now();
    vector_add<<<dim3(blocks), dim3(threads)>>>(cu_arr1, cu_arr2);
    cudaDeviceSynchronize();
    auto timer_stop = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::nanoseconds>(timer_stop - timer_start);
    cout << threads << ";" << duration.count() << endl;

    if (cudaMemcpy(result, cu_arr1, N * sizeof(float), cudaMemcpyDeviceToHost))
    {
      cerr << "Error when copy memory from device to host";
      return -1;
    }

    for (size_t i = 0; i < N; i++)
    {
      if (result[i] != arr1[i] + arr2[i])
      {
        cout << "result[i] != arr1[i] + arr2[i]" << endl;
        return -1;
      }
    }

    cudaFree(&cu_arr1);
    cudaFree(&cu_arr2);

    delete[] arr1;
    delete[] arr2;
    delete[] result;
  }

  return 0;
}