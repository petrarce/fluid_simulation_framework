#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;
out vec4 newColor;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    newColor = vec4(aPos, 1.0);
}