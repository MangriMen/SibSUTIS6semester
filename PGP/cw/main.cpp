#include "opengl.hpp"
#include <GLFW/glfw3.h>
#include <iostream>
#include "opengl_cs.hpp"

using namespace std;

const unsigned int window_width = 512;
const unsigned int window_height = 512;

void initGL();

GLuint renderHandle, computeHandle;

void updateTex(int);
void draw();

GLFWwindow *window;

int main() {
	initGL();

	GLuint texHandle = genTexture();
	renderHandle = genRenderProg(texHandle);
	computeHandle = genComputeProg(texHandle);

	for (int i = 0; i < 1024; ++i) {
		updateTex(i);
		draw();
	}

	return 0;
}

void initGL()
{
	if (!glfwInit())
	{
		cerr << "Failed to initialize GLFW"
			 << "\n";
		(void)getchar();
		return;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	window = glfwCreateWindow(window_width, window_height, "Window", NULL, NULL);

	if (window == NULL)
	{
		cerr << "Failed to open GLFW window."
			 << "\n";
		(void)getchar();
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true;
	if (glewInit() != GLEW_OK)
	{
		cerr << "Failed to initialize GLEW"
			 << "\n";
		(void)getchar();
		glfwTerminate();
		return;
	}
}

void checkErrors(std::string desc) {
	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		fprintf(stderr, "OpenGL error in \"%s\": %s (%d)\n", desc.c_str(),
			gluErrorString(e), e);
		exit(20);
	}
}

void updateTex(int frame) {
	glUseProgram(computeHandle);
	glUniform1f(glGetUniformLocation(computeHandle, "roll"), (float)frame * 0.01f);
	glDispatchCompute(512 / 16, 512 / 16, 1); // 512^2 threads in blocks of 16^2
	checkErrors("Dispatch compute shader");
}

void draw() {
	glUseProgram(renderHandle);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glfwSwapBuffers(window);
	checkErrors("Draw screen");
}