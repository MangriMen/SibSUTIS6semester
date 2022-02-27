#include <iostream>
#include <iomanip>
#include <fstream>
#include <cufft.h>

using namespace std;

#define NX 365
#define BATCH 1

int main()
{
    cufftHandle plan;
    cufftComplex *data;

    cufftComplex *data_h = new cufftComplex[NX * BATCH];

    ifstream fileIn("w1991.dat");

    if (!fileIn.is_open())
    {
        cerr << "Failed to open in file." << endl;
        return EXIT_FAILURE;
    }

    float month = 0;
    float day = 0;
    float wolf = 0;
    float temp = 0;
    for (int i = 0; i < NX; i++)
    {
        fileIn >> month >> day >> wolf >> temp;
        if (wolf != 999)
        {
            data_h[i].x = wolf;
        }
        else
        {
            data_h[i].x = 0.0f;
        }

        data_h[i].y = 0.0f;
    }

    fileIn.close();

    cudaMalloc((void **)&data, sizeof(cufftComplex) * NX * BATCH);
    cudaMemcpy(data, data_h, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyHostToDevice);

    if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS)
    {
        cerr << "CUFFT error: Plan creation failed." << endl;
        return -1;
    }
    if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        cerr << "CUFFT error: ExecC2C Forward failed." << endl;
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(data_h, data, NX * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    ofstream fileOut("out.dat");
    if (!fileOut.is_open())
    {
        cerr << "Failed to open out file." << endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < NX; i++)
    {
        cout << setw(-20) << data_h[i].x << setw(20) << data_h[i].y << endl;
        fileOut << data_h[i].x << ";" << data_h[i].y << endl;
    }

    fileOut.close();

    cufftDestroy(plan);
    cudaFree(data);
    delete[] data_h;

    return EXIT_SUCCESS;
}