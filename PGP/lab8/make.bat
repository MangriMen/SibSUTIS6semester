call "C:\Programs\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc lab8_thrust.cu -o lab8_thrust.exe
nvcc lab8_cublas.cu -l cublas -o lab8_cublas.exe