#pragma once

#include <iostream>
#include <glad.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


using namespace std;

class texture
{
private:
	unsigned int texId;
	GLenum texType;
	GLint texBehaviour;
	GLint texFilter;
	GLint texNumColorComp;
	GLenum texFormat;
	GLenum texBinding;
public:
	opcode setTexBehaviout(GLint texBehaviour)
	{
		unsigned int curBoundTexture;
		glGetIntegerv(texBinding, (int*)&curBoundTexture);
		glBindTexture(texType, this->texId);
		this->texBehaviour = texBehaviour;
	    glTexParameteri(texType, GL_TEXTURE_WRAP_S, texBehaviour);
	    glTexParameteri(texType, GL_TEXTURE_WRAP_T, texBehaviour);
	    glBindTexture(texType, curBoundTexture);
	    return STATUS_OK;
	};

	opcode setTexFilter(GLint texFilter)
	{
		unsigned int curBoundTexture;
		glGetIntegerv(texBinding, (int*)&curBoundTexture);
		glBindTexture(texType, this->texId);
		this->texFilter = texFilter;
	    glTexParameteri(texType, GL_TEXTURE_MIN_FILTER, texFilter);
	    glTexParameteri(texType, GL_TEXTURE_MAG_FILTER, texFilter);
	    glBindTexture(texType, curBoundTexture);
	    return STATUS_OK;
	};

	opcode use()
	{
		glBindTexture(this->texType, this->texId);
		return STATUS_OK;
	};

	texture(string pathToTexture, 
			GLenum texType,
			GLint texBehaviour,
			GLint texFilter,
			GLint texNumColorComp,
			GLenum texFormat):
				texType(texType),
				texBehaviour(texBehaviour),
				texFilter(texFilter),
				texNumColorComp(texNumColorComp),
				texFormat(texFormat)	
	{

	    glGenTextures(1, &this->texId);
	    glBindTexture(texType, this->texId);
	    switch(texType){
	    	case GL_TEXTURE_2D:
	    		texBinding = GL_TEXTURE_BINDING_2D;
	    		break;
	    	default:
	    		exit(-1);
	    		break;
	    }
	    //glTexParameteri(texType, GL_TEXTURE_WRAP_S, texBehaviour);
	    //glTexParameteri(texType, GL_TEXTURE_WRAP_T, texBehaviour);
	    setTexBehaviout(texBehaviour);
	    //glTexParameteri(texType, GL_TEXTURE_MIN_FILTER, texFilter);
	    //glTexParameteri(texType, GL_TEXTURE_MAG_FILTER, texFilter);
	    setTexFilter(texFilter);
	    int wdth, hgh, nchan;
	    uint8_t* texData = stbi_load(pathToTexture.c_str(), &wdth, &hgh, &nchan, 0);
	    if(!texData){
	        printf("failed to load texture image");
	        exit(-1);
	    } else {
	        glTexImage2D(texType, 0, texNumColorComp, wdth, hgh, 0, texFormat, GL_UNSIGNED_BYTE, texData);
	        glGenerateMipmap(texType);
	    }
	    stbi_image_free(texData);
	};
};