#include <GL/glew.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

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

const unsigned int window_width = 512;
const unsigned int window_height = 512;
GLuint bufferID;
GLuint progHandle;
GLuint genRenderProg();
const int num_of_verticies = 3;

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
        0.0f, //
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.9f, //
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
    glVertexAttribPointer(colorPtr, 3, GL_FLOAT, GL_FALSE, 24, (const GLvoid *)12);
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