call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc native_saxpy.cu -o native_saxpy.exe
nvcc thrust_saxpy.cu -o thrust_saxpy.exe
nvcc cublas_saxpy.cu -lcublas -o cublas_saxpy.exe