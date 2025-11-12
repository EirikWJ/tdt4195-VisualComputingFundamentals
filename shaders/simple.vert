#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 normals;

uniform mat4 transform;
uniform mat4 model;


out vec4 vcolor;
out vec3 vnormals;

void main()
{
    gl_Position = transform * vec4(position, 1.0f);

    vcolor = color;

    vnormals = normalize(mat3(model) * normals);
}