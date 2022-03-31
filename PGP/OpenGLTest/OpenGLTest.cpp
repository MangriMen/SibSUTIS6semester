#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>

using namespace std;

#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <fstream>
void initGL();
int initBuffer();
void display();
void myCleanup();
GLFWwindow* window;
const unsigned int window_width = 512;
const unsigned int window_height = 512;

GLuint bufferID;
GLuint progHandle;
GLuint genRenderProg();
const int num_of_verticies = 3;

GLuint genComputeProgram();
void compute();

int main()
{
	initGL();
	initBuffer();

	compute();

	//do
	//{
	//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//	display();
	//	glfwSwapBuffers(window);
	//	glfwPollEvents();
	//} while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	myCleanup();

	glfwTerminate();
	return 0;
}

void initGL()
{
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE,
		GLFW_OPENGL_COMPAT_PROFILE);
	window = glfwCreateWindow(window_width, window_height,
		"Template window", NULL, NULL);
	if (window == NULL)
	{
		fprintf(stderr, "Failed to open GLFW window. \n");
		getchar();
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(window);
	// Initialize GLEW
	glewExperimental = true;
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return;
	}
	return;
}

void checkErrors(std::string desc)
{
	GLenum e = glGetError();
	if (e != GL_NO_ERROR)
	{
		fprintf(stderr, "OpenGL error in \"%s\": %s (%d)\n", desc.c_str(),
			gluErrorString(e), e);
		exit(20);
	}
}

int initBuffer()
{
	glGenBuffers(1, &bufferID);
	glBindBuffer(GL_ARRAY_BUFFER, bufferID);
	static const GLfloat vertex_buffer_data[] = {
		-0.9f,
		-0.9f,
		-0.0f,
		1.0f,
		0.0f,
		0.0f,
		0.0f,
		0.0f,
		0.0f,
		0.0f,
		1.0f,
		0.0f,
		0.9f,
		-0.5f,
		0.0f,
		0.0f,
		0.0f,
		1.0f,
	};

	glBufferData(GL_ARRAY_BUFFER, 6 * num_of_verticies * sizeof(float),
		vertex_buffer_data, GL_STATIC_DRAW);

	return 0;
}

void display()
{
	progHandle = genRenderProg();
	glUseProgram(progHandle);

	GLint posPtr = glGetAttribLocation(progHandle, "pos");
	glVertexAttribPointer(posPtr, 3, GL_FLOAT, GL_FALSE, 24, 0);
	glEnableVertexAttribArray(posPtr);
	GLint colorPtr = glGetAttribLocation(progHandle, "color");
	glVertexAttribPointer(colorPtr, 3, GL_FLOAT, GL_FALSE, 24, (const GLvoid*)12);
	glEnableVertexAttribArray(colorPtr);

	glDrawArrays(GL_TRIANGLES, 0, num_of_verticies);

	glDisableVertexAttribArray(posPtr);
	glDisableVertexAttribArray(colorPtr);
}

void myCleanup()
{
	glDeleteBuffers(1, &bufferID);
	glDeleteProgram(progHandle);
}

GLuint genRenderProg()
{
	GLuint progHandle = glCreateProgram();
	GLuint vp = glCreateShader(GL_VERTEX_SHADER);
	GLuint fp = glCreateShader(GL_FRAGMENT_SHADER);

	const char* vpSrc[] = {
		"#version 430\n",
		"layout(location = 0) in vec3 pos;\
		layout(location = 1) in vec3 color;\
out vec4 vs_color;\
void main() {\
 gl_Position = vec4(pos,1);\
 vs_color=vec4(color,1.0);\
}" };
	const char* fpSrc[] = {
		"#version 430\n",
		"in vec4 vs_color;\
 out vec4 fcolor;\
 void main() {\
fcolor = vs_color;\
}" };
	glShaderSource(vp, 2, vpSrc, NULL);
	glShaderSource(fp, 2, fpSrc, NULL);
	glCompileShader(vp);

	int rvalue;
	glGetShaderiv(vp, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue)
	{
		fprintf(stderr, "Error in compiling vp\n");
		exit(30);
	}
	glAttachShader(progHandle, vp);

	glCompileShader(fp);
	glGetShaderiv(fp, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue)
	{
		fprintf(stderr, "Error in compiling fp\n");
		exit(31);
	}
	glAttachShader(progHandle, fp);
	glLinkProgram(progHandle);
	glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
	if (!rvalue)
	{
		fprintf(stderr, "Error in linking sp\n");
		exit(32);
	}
	checkErrors("Render shaders");
	return progHandle;
}

struct CharString {
	const char* p;
	CharString(const std::string& s) : p(s.c_str()) {}
	operator const char** () { return &p; }
};

const string getShaderFromFile(const string filename) {
	
	ifstream fileIn(filename);
	if (!fileIn.is_open()) {
		cerr << "Error opening " << filename << "\n";
		throw new runtime_error("Error openning file");
	}

	stringstream buffer;
	buffer << fileIn.rdbuf();
	
	fileIn.close();

	return buffer.str();
}

GLuint compileShader(const string shaderStr, int shaderType) {
	GLuint shader = glCreateShader(shaderType);
	
	glShaderSource(shader, 1, CharString(shaderStr), NULL);
	glCompileShader(shader);
	
	int status = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (!status)
	{
		fprintf(stderr, "Error in compiling computeShader\n");
		exit(30);
	}

	return shader;
}

void linkProgram(GLuint program) {
	glLinkProgram(program);

	int status = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (!status)
	{
		fprintf(stderr, "Error in linking program %lu\n", program);
		exit(32);
	}
}

string arrayGenComputeShaderString;
string saxpyComputeShaderString = getShaderFromFile("saxpy.comp");

GLuint genComputeProgram()
{
	GLuint computeProgram = glCreateProgram();

	GLuint computeShader = compileShader(saxpyComputeShaderString, GL_COMPUTE_SHADER);
	glAttachShader(computeProgram, computeShader);

	linkProgram(computeProgram);
	checkErrors("Render shaders");
	return computeProgram;
}

void compute()
{
	GLuint computeProgram = genComputeProgram();

	glUseProgram(computeProgram);
	glDispatchCompute(1 << 24, 1, 1);
	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	glDeleteProgram(computeProgram);
}