#version 330 core
out vec4 FragColor;
in vec4 newColor;
void main()
{
	FragColor = newColor.xyzw;
}
