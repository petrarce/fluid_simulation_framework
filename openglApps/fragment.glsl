#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
in vec4 newColor;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform float textureMixRate;

void main()
{
    //FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2);
    FragColor = mix(texture(texture1, TexCoord), texture(texture2, vec2(1 - TexCoord[0], TexCoord[1])), 
    				textureMixRate);
}