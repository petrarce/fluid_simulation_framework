#include <glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <marching_cubes.h>
#include <shader.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace glm;
using namespace std;
using namespace learnSPH;
using namespace Eigen;

int wiewWidth = 500, wiewHeight = 500;
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    wiewWidth = width;
    wiewHeight = height;
}  

void processInput(GLFWwindow* window)
{
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
		glfwSetWindowShouldClose(window, true);
	}
}

int main(int argc, char** argv)
{


    assert(argc == 6);
    Vector3R lower_corner = {-1, -1, -1};
    Vector3R upper_corner = {1, 1, 1};
    Vector3R cubeResolution = {0.1, 0.1, 0.1};
    Vector3R sphereCenter = {0, 0, 0};
    Real sphereRadius = 0.5;
    string shaders_dir = string(argv[1]);
    string texturePath1 = string(argv[2]);
    string texturePath2 = string(argv[3]);
    Real textureMixRate = stod(argv[4]);

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  
    GLFWwindow* window = glfwCreateWindow(wiewWidth, wiewHeight, "tutorial 1", NULL, NULL);
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);  
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
    Matrix4d transformationMatr = Matrix4d::Identity();
    mat4 scl    = scale(mat4(1), vec3(1, 1, 1));
    mat4 rot;
    mat4 transformMatr;
    glUniform1i(glGetUniformLocation(myShaiders.getId(), "texture1"), 0);
    glUniform1i(glGetUniformLocation(myShaiders.getId(), "texture2"), 1);
    glUniform1f(glGetUniformLocation(myShaiders.getId(), "textureMixRate"), textureMixRate);
    glEnable(GL_DEPTH_TEST);

    //unbind VAO and VBO
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

    float grad = 0;
    int i = 0;
    glfwSetTime(0);
    int frames = 0;
    while(!glfwWindowShouldClose(window)){
        i++;
        frames++;
    	processInput(window);

    	glClearColor( 	0, 0, 0, 0.3);
    	glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //glBindTexture(GL_TEXTURE_2D, texture);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);
        glBindVertexArray(VAO);

        myShaiders.use();

        mat4 view = lookAt(vec3(0,0,3+1*sin(radians(float(i)))), vec3(0,0,0), vec3(0,1,0));
        glUniformMatrix4fv(glGetUniformLocation(myShaiders.getId(), "view"), 
                            1, 
                            GL_FALSE, 
                            (float*)&view);
        mat4 proj = perspective(radians(45.0f), float(wiewWidth)/wiewHeight, 0.1f, 100.0f);
        glUniformMatrix4fv(glGetUniformLocation(myShaiders.getId(), "perspective"), 
                            1, 
                            GL_FALSE, 
                            (float*)&proj);
        grad = int(grad + 1)%360;
        rot    = rotate(mat4(1), radians(grad), vec3(1,1,1));
        mat4 trans;
/*
        trans = translate(mat4(1.0f), vec3(0,0,0));
        transformMatr = trans * (scl * rot);
        glUniformMatrix4fv(glGetUniformLocation(myShaiders.getId(), "transform"), 
                            1, 
                            GL_FALSE, 
                            (float*)&transformMatr);
        glDrawArrays(GL_TRIANGLES, 0, triangle_mesh.size());

*/
       int maxObj = stoi(argv[5]);
        for(int j = 0; j < sqrt(maxObj); j++){
            for(int k = 0; k < sqrt(maxObj); k++){
                trans = translate(mat4(1.0f), vec3(j/sqrt(maxObj),k/sqrt(maxObj),0));
                transformMatr = trans * (scl * rot);
                glUniformMatrix4fv(glGetUniformLocation(myShaiders.getId(), "transform"), 
                                    1, 
                                    GL_FALSE, 
                                    (float*)&transformMatr);
                glDrawArrays(GL_TRIANGLES, 0, triangle_mesh.size());
            }
        }


    	glfwPollEvents();
    	glfwSwapBuffers(window);
        if(glfwGetTime() > 1){
            pr_dbg("frame rate: %d fps", frames);
            glfwSetTime(0);
            frames = 0;
        }
    }

    glfwTerminate();
    return 0;
}

