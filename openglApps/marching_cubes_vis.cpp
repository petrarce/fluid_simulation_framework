#include <glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <marching_cubes.h>
#include <shader.h>

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


    assert(argc == 15);
    Vector3R lower_corner = {stod(argv[1]), stod(argv[2]), stod(argv[3])};
    Vector3R upper_corner = {stod(argv[4]), stod(argv[5]), stod(argv[6])};
    Vector3R cubeResolution = {stod(argv[7]), stod(argv[8]), stod(argv[9])};
    Vector3R sphereCenter = {stod(argv[10]), stod(argv[11]), stod(argv[12])};
    Real sphereRadius = stod(argv[13]);
    string shaders_dir = string(argv[14]);

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

    vector<Vector3R> triangle_mesh;
    Sphere sphr(sphereRadius, sphereCenter);
    MarchingCubes mcb;
    mcb.init(lower_corner, upper_corner, cubeResolution);
    mcb.setObject(&sphr);
    mcb.getTriangleMesh(triangle_mesh);

    unsigned int VBO, VAO;

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vector3R)*triangle_mesh.size(), triangle_mesh.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, sizeof(Vector3R), (void*)0);
	glEnableVertexAttribArray(0);


	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

    while(!glfwWindowShouldClose(window)){
    	processInput(window);

    	glClearColor( 	0, 0, 0, 0.3);
    	glClear(GL_COLOR_BUFFER_BIT);

    	myShaiders.use();
    	glBindVertexArray(VAO);
    	glDrawArrays(GL_TRIANGLES, 0, triangle_mesh.size());

    	glfwPollEvents();
    	glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}