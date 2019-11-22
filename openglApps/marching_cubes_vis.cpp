#include <glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <marching_cubes.h>
#include <shader.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace std;
using namespace learnSPH;


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}  

void processInput(GLFWwindow* window)
{
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
		glfwSetWindowShouldClose(window, true);
	}
}

int main(int argc, char** argv)
{


    assert(argc == 18);
    Vector3R lower_corner = {stod(argv[1]), stod(argv[2]), stod(argv[3])};
    Vector3R upper_corner = {stod(argv[4]), stod(argv[5]), stod(argv[6])};
    Vector3R cubeResolution = {stod(argv[7]), stod(argv[8]), stod(argv[9])};
    Vector3R sphereCenter = {stod(argv[10]), stod(argv[11]), stod(argv[12])};
    Real sphereRadius = stod(argv[13]);
    string shaders_dir = string(argv[14]);
    string texturePath1 = string(argv[15]);
    string texturePath2 = string(argv[16]);
    Real textureMixRate = stod(argv[17]);

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  
    GLFWwindow* window = glfwCreateWindow(500, 500, "tutorial 1", NULL, NULL);
    if(window == NULL){
    	cout << "failed to create glfw window" << endl;
    	glfwTerminate();
    	return -1;
    }

    glfwMakeContextCurrent(window);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
    	cout << "failed to load opengl functions" << endl;
    	return -1;
    }

    glViewport(0,0,500,500);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  


    //add shader here
    Shader myShaiders(shaders_dir + "vertex.glsl", shaders_dir + "fragment.glsl");

    vector<Real> data_array;
    vector<Vector3R> triangle_mesh;
    Thorus sphr(sphereRadius, sphereRadius/2, sphereCenter);
    MarchingCubes mcb;
    mcb.init(lower_corner, upper_corner, cubeResolution);
    mcb.setObject(&sphr);
    mcb.getTriangleMesh(triangle_mesh);

    for(const Vector3R& pt : triangle_mesh)
    {
        data_array.push_back(pt(0));
        data_array.push_back(pt(1));
        data_array.push_back(pt(2));
        data_array.push_back((pt(0) + 1)/2);
        data_array.push_back((pt(1) + 1)/2);
    }
/*
    data_array = {  -0.5,       -0.5,   0, 0, 0,
                    -0.5,       0.5,    0, 0, 1,
                    0.5,        0.5,    0, 1, 1,
                    -0.5,       -0.5,   0, 0, 0,
                    0.5,        0.5,    0, 1, 1,
                    0.5,        -0.5,   0, 1, 0
            };
*/

    unsigned int VBO, VAO;//, EBO;

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
    //glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(data_array[0])*data_array.size(), data_array.data(), GL_STATIC_DRAW);

    /*glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indeces), indeces, GL_STATIC_DRAW);*/

    int stride = sizeof(data_array[0])*5;
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, stride, (void*)0);
	glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_DOUBLE, GL_FALSE, stride, (void*)(sizeof(data_array[0])*3));
    glEnableVertexAttribArray(1);

    unsigned int texture1, texture2;
    glGenTextures(1, &texture1);
    //glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    int wdth, hgh, nchan;
    uint8_t* texData = stbi_load(texturePath1.c_str(), &wdth, &hgh, &nchan, 0);
    if(!texData){
        printf("failed to load texture image");
        return -1;
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, wdth, hgh, 0, GL_RGB, GL_UNSIGNED_BYTE, texData);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    stbi_image_free(texData);

    glGenTextures(1, &texture2);
    //glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   // set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  
    stbi_set_flip_vertically_on_load(true);      
    texData = stbi_load(texturePath2.c_str(), &wdth, &hgh, &nchan, 0);
    if(!texData){
        printf("failed to load texture image");
        return -1;
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, wdth, hgh, 0, GL_RGBA, GL_UNSIGNED_BYTE, texData);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    stbi_image_free(texData);

    myShaiders.use();
    glUniform1i(glGetUniformLocation(myShaiders.getId(), "texture1"), 0);
    glUniform1i(glGetUniformLocation(myShaiders.getId(), "texture2"), 1);
    glUniform1f(glGetUniformLocation(myShaiders.getId(), "textureMixRate"), textureMixRate);
    //unbind VAO and VBO
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

    while(!glfwWindowShouldClose(window)){
    	processInput(window);

    	glClearColor( 	0, 0, 0, 0.3);
    	glClear(GL_COLOR_BUFFER_BIT);

        //glBindTexture(GL_TEXTURE_2D, texture);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);
    	myShaiders.use();
    	glBindVertexArray(VAO);
    	glDrawArrays(GL_TRIANGLES, 0, triangle_mesh.size());

    	glfwPollEvents();
    	glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}

