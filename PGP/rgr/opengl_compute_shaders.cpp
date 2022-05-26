#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>

using namespace std;

constexpr unsigned int WINDOW_WIDTH = 512;
constexpr unsigned int WINDOW_HEIGHT = 512;

constexpr unsigned short OPENGL_MAJOR_VERSION = 4;
constexpr unsigned short OPENGL_MINOR_VERSION = 3;

GLfloat vertices[] =
	{
		-1.0f,
		-1.0f,
		0.0f,
		0.0f,
		0.0f,
		-1.0f,
		1.0f,
		0.0f,
		0.0f,
		1.0f,
		1.0f,
		1.0f,
		0.0f,
		1.0f,
		1.0f,
		1.0f,
		-1.0f,
		0.0f,
		1.0f,
		0.0f,
};

GLuint indices[] =
	{
		0, 2, 1,
		0, 3, 2};

const char *screenVertexShaderSource = // Vertex Shader
	R"(
	#version 430 core
	layout (location = 0) in vec3 pos;
	layout (location = 1) in vec2 uvs;
	out vec2 UVs;
	void main()
	{
		gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);
		UVs = uvs;
	}
)";
const char *screenFragmentShaderSource = // Fragment Shader
	R"(
	#version 430 core
	out vec4 FragColor;
	uniform sampler2D screen;
	in vec2 UVs;
	void main()
	{
		FragColor = texture(screen, UVs);
	}
)";
const char *saxpyComputeShaderSource = // Compute Shader
	R"(
	#version 430 core
	layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
	layout(rgba32f, binding = 0) uniform image2D screen;
	void main()
	{
		ivec2 dims = imageSize(screen);

		uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * dims.x;

		float value = 2.0f * i + 0;

		//float fromEnd = dims.x * dims.y;
		//float toEnd = 1;

		//float grad = value / fromEnd * toEnd;

		//vec4 pixel = vec4(grad, grad, grad, 1.0);
		//ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

		//imageStore(screen, pixel_coords, pixel);
	}
)";

int main()
{
	int n = 1 << 24;

	if (argc > 1)
	{
		n = stoi(argv[1]);
	}
	else
	{
		return EXIT_FAILURE;
	}

	chrono::steady_clock::time_point startGlobal, endGlobal, startLocal, endLocal;

	startGlobal = chrono::high_resolution_clock::now();

	// Init
	if (!glfwInit())
	{
		cerr << "Failed to initialize GLFW"
			 << "\n";
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_MAJOR_VERSION);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_MINOR_VERSION);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Compute shader", NULL, NULL);

	if (window == NULL)
	{
		cerr << "Failed to open GLFW window"
			 << "\n";
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);

	glewExperimental = true;
	if (glewInit() != GLEW_OK)
	{
		cerr << "Failed to initialize GLEW"
			 << "\n";
		glfwTerminate();
	}

	// VAO, VBO, EBO
	GLuint VAO, VBO, EBO;
	glCreateVertexArrays(1, &VAO);
	glCreateBuffers(1, &VBO);
	glCreateBuffers(1, &EBO);

	glNamedBufferData(VBO, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glNamedBufferData(EBO, sizeof(indices), indices, GL_STATIC_DRAW);

	glEnableVertexArrayAttrib(VAO, 0);
	glVertexArrayAttribBinding(VAO, 0, 0);
	glVertexArrayAttribFormat(VAO, 0, 3, GL_FLOAT, GL_FALSE, 0);

	glEnableVertexArrayAttrib(VAO, 1);
	glVertexArrayAttribBinding(VAO, 1, 0);
	glVertexArrayAttribFormat(VAO, 1, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat));

	glVertexArrayVertexBuffer(VAO, 0, VBO, 0, 5 * sizeof(GLfloat));
	glVertexArrayElementBuffer(VAO, EBO);

	// Texture
	GLuint screenTex;
	glCreateTextures(GL_TEXTURE_2D, 1, &screenTex);
	glTextureParameteri(screenTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri(screenTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTextureParameteri(screenTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(screenTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTextureStorage2D(screenTex, 1, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT);
	glBindImageTexture(0, screenTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

	// Shaders
	GLuint screenVertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(screenVertexShader, 1, &screenVertexShaderSource, NULL);
	glCompileShader(screenVertexShader);
	GLuint screenFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(screenFragmentShader, 1, &screenFragmentShaderSource, NULL);
	glCompileShader(screenFragmentShader);

	GLuint screenShaderProgram = glCreateProgram();
	glAttachShader(screenShaderProgram, screenVertexShader);
	glAttachShader(screenShaderProgram, screenFragmentShader);
	glLinkProgram(screenShaderProgram);

	glDeleteShader(screenVertexShader);
	glDeleteShader(screenFragmentShader);

	GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(computeShader, 1, &saxpyComputeShaderSource, NULL);
	glCompileShader(computeShader);

	GLuint computeProgram = glCreateProgram();
	glAttachShader(computeProgram, computeShader);
	glLinkProgram(computeProgram);

	glDeleteShader(computeShader);

	// do
	//{
	//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(computeProgram);
	startLocal = chrono::high_resolution_clock::now();
	glDispatchCompute(n, 1, 1);
	glMemoryBarrier(GL_ALL_BARRIER_BITS);
	endLocal = chrono::high_resolution_clock::now();

	//	glUseProgram(screenShaderProgram);
	//	glBindTextureUnit(0, screenTex);
	//	glUniform1i(glGetUniformLocation(screenShaderProgram, "screen"), 0);
	//	glBindVertexArray(VAO);
	//	glDrawElements(GL_TRIANGLES, sizeof(indices) / sizeof(indices[0]), GL_UNSIGNED_INT, 0);

	//	glfwSwapBuffers(window);
	//	glfwPollEvents();
	//} while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	glfwDestroyWindow(window);
	glfwTerminate();

	endGlobal = chrono::high_resolution_clock::now();

	long long globalElapsedTime = chrono::duration_cast<chrono::milliseconds>(endGlobal - startGlobal).count();

	long long localElapsedTime = chrono::duration_cast<chrono::microseconds>(endLocal - startLocal).count();

	printf("OpenGL\n\tGlobal: %lld ms\n\tLocal: %f ms\n", globalElapsedTime, localElapsedTime / (double)1000);

	ofstream fileOut("result.csv", ios::app);
	fileOut << "opengl;" << globalElapsedTime << ";" << localElapsedTime / (double)1000 << "\n";
	fileOut.close();

	return EXIT_SUCCESS;
}