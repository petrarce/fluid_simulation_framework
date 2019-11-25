#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;


out vec2 TexCoord;
out vec4 newColor;

uniform mat4 perspective;
uniform mat4 world;
uniform mat4 view;


void main()
{
    gl_Position = perspective * view *  (world * vec4(aPos, 1.0));
    newColor = vec4((aPos + vec3(1,1,1)) / 2, 1.0);
    TexCoord = aTexCoord;
}