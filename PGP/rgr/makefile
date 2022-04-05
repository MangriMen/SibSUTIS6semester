.PHONY: run debug clear rgr test

CXX=cl
CXXFLAGS=/std:c++17 /EHsc /MT /O2
LINKER_FLAGS=/SUBSYSTEM:windows /ENTRY:mainCRTStartup /NODEFAULTLIB:LIBCMT
DEBUG_FLAGS=/DEBUG:FULL /Zi

DIR_INCLUDE="C:\Tools\msvc2022\include"
DIR_LIB_x86="C:\Tools\msvc2022\lib\x86"
DIR_LIB_x64="C:\Tools\msvc2022\lib\x64"

DIR_RELEASE=bin\Release
DIR_DEBUG=bin\Debug

LIBS=opengl32.lib glew32s.lib glfw3.lib glu32.lib
WINDOWS_LIBS=user32.lib gdi32.lib shell32.lib vcruntime.lib msvcrt.lib

SOURCES=opengl_compute_shaders.cpp
NAME=opengl_compute_shaders.exe

!if "$(VSCMD_ARG_HOST_ARCH)" == "x64"
DIR_LIB=$(DIR_LIB_x64)
DIR_BIN_RELEASE=$(DIR_RELEASE)\x64
DIR_BIN_DEBUG=$(DIR_DEBUG)\x64
!else
DIR_LIB=$(DIR_LIB_x86)
DIR_BIN_RELEASE=$(DIR_RELEASE)\x86
DIR_BIN_DEBUG=$(DIR_DEBUG)\x86
!endif

run: $(DIR_BIN_RELEASE)\$(NAME)
	start /b .\$(DIR_BIN_RELEASE)\$(NAME)

debug: $(DIR_BIN_DEBUG)\$(NAME)
	start /b .\$(DIR_BIN_DEBUG)\$(NAME)

clear:
	del *.obj

rgr:
	nvcc native_saxpy.cu -o native_saxpy.exe
	nvcc thrust_saxpy.cu -o thrust_saxpy.exe
	nvcc cublas_saxpy.cu -lcublas -o cublas_saxpy.exe
	$(CXX) $(CXXFLAGS) $(SOURCES) /I $(DIR_INCLUDE) /Fe:$(NAME) /link $(LINKER_FLAGS) /LIBPATH:$(DIR_LIB) $(LIBS) $(WINDOWS_LIBS)

test:
	del result.csv
	.\native_saxpy.exe
	.\thrust_saxpy.exe
	.\cublas_saxpy.exe
	.\$(NAME)

$(DIR_BIN_RELEASE)\$(NAME):
	if not exist $(DIR_BIN_RELEASE) mkdir $(DIR_BIN_RELEASE)
	$(CXX) $(CXXFLAGS) $(SOURCES) /I $(DIR_INCLUDE) /Fo$(DIR_BIN_RELEASE)\ /Fe:$(DIR_BIN_RELEASE)\$(NAME) /link $(LINKER_FLAGS) /LIBPATH:$(DIR_LIB) $(LIBS) $(WINDOWS_LIBS)

$(DIR_BIN_DEBUG)\$(NAME):
	if not exist $(DIR_BIN_DEBUG) mkdir $(DIR_BIN_DEBUG)
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) /Fd$(DIR_BIN_DEBUG)\ /Fo$(DIR_BIN_DEBUG)\ $(SOURCES) /I $(DIR_INCLUDE) /Fe:$(DIR_BIN_DEBUG)\$(NAME) /link $(LINKER_FLAGS) /LIBPATH:$(DIR_LIB) $(LIBS) $(WINDOWS_LIBS)
