#version 330 core
layout (location = 0) in vec3 aPos;
out vec4 newColor;
void main()
{
   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
   newColor = vec4(aPos.xyz, 0.1f);
}
