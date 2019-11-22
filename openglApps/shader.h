#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <glad.h>

using namespace std;

class Shader {
private:
	unsigned int prog_id;

public:
	Shader(string vertexPath, string fragmentPath)
	{
  		string vertexCode;
  		string fragmentCode;
  		ifstream vertexFile;
  		ifstream fragmentFile;

  		try{

  			vertexFile.open(vertexPath);
  			fragmentFile.open(fragmentPath);

  			stringstream vertexStrStream;
  			stringstream fragmentStrStream;

  			vertexStrStream << vertexFile.rdbuf();
  			fragmentStrStream << fragmentFile.rdbuf();

  			vertexFile.close();
  			fragmentFile.close();

  			vertexCode = vertexStrStream.str();
  			fragmentCode = fragmentStrStream.str();
  		} catch(const std::ifstream::failure& e){
  			printf("failed to read a sader sorce file\n");
  			return;
  		}

  		const char* verShaderPtr = vertexCode.c_str();
  		const char* fragShaderPtr = fragmentCode.c_str();
  		unsigned int vertex, fragment;
  		int success;
  		char infoLog[512];

  		vertex = glCreateShader(GL_VERTEX_SHADER);
  		glShaderSource(vertex, 1, &verShaderPtr, NULL);
  		glCompileShader(vertex);
  		glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
  		if(!success){
  			glGetShaderInfoLog(vertex, 512, NULL, infoLog);
  			printf("VERTEX SHADER COMPILTION ERRORS: %s\n", infoLog);
  			return;
  		}

  		fragment = glCreateShader(GL_FRAGMENT_SHADER);
  		glShaderSource(fragment, 1, &fragShaderPtr, NULL);
  		glCompileShader(fragment);
  		glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
  		if(!success){
  			glGetShaderInfoLog(fragment, 512, NULL, infoLog);
  			printf("FRAGMENT SHADER COMPILTION ERRORS: %s\n", infoLog);
  			return;
  		}

  		this->prog_id = glCreateProgram();
  		glAttachShader(this->prog_id, vertex);
  		glAttachShader(this->prog_id, fragment);
  		glLinkProgram(this->prog_id);

  		glGetProgramiv(this->prog_id, GL_LINK_STATUS, &success);
  		if(!success){
  			glGetProgramInfoLog(this->prog_id, 512, NULL, infoLog);
   			printf("Falied to link final shaders: %s\n", infoLog);
  			return;
 		}

 		glDeleteShader(vertex);
 		glDeleteShader(fragment);

	};
	~Shader()
	{
		
	}
	void use()
	{
		glUseProgram(this->prog_id);
	};

  unsigned int getId(){return prog_id;};

};