#include <glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <marching_cubes.h>
#include <shader.h>
#include <texture.hpp>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace glm;
using namespace std;
using namespace learnSPH;
using namespace Eigen;

int wiewWidth = 500, wiewHeight = 500;

bool firstMouse = true;
float yawV   = -90.0f;   // yawV is initialized to -90.0 degrees since a yawV of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
float pitchV =  0.0f;
float lastX =  800.0f / 2.0;
float lastY =  600.0 / 2.0;
float fov   =  45.0f;

glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);

float deltaTime = 0.01f; // time between current frame and last frame
float lastFrame = 0.0f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    wiewWidth = width;
    wiewHeight = height;
}  

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f; // change this value to your liking
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yawV += xoffset;
    pitchV += yoffset;

    // make sure that when pitchV is out of bounds, screen doesn't get flipped
    if (pitchV > 89.0f)
        pitchV = 89.0f;
    if (pitchV < -89.0f)
        pitchV = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yawV)) * cos(glm::radians(pitchV));
    front.y = sin(glm::radians(pitchV));
    front.z = sin(glm::radians(yawV)) * cos(glm::radians(pitchV));
    cameraFront = glm::normalize(front);

}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = 2.5 * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
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

    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);


    //add shader here
    Shader myShaiders(shaders_dir + "vertex.glsl", shaders_dir + "fragment.glsl");

    vector<Real> data_array;
    vector<Vector3R> triangle_mesh;
    Thorus thr(sphereRadius, 0.5*sphereRadius, sphereCenter, lower_corner, upper_corner, cubeResolution);;
    MarchingCubes mcb;
    mcb.init(lower_corner, upper_corner, cubeResolution);
    mcb.setObject(&thr);
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


    texture tex1(texturePath1, GL_TEXTURE_2D, GL_REPEAT, GL_LINEAR, GL_RGB, GL_RGB);
    texture tex2(texturePath2, GL_TEXTURE_2D, GL_REPEAT, GL_NEAREST, GL_RGBA, GL_RGBA);
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
        tex1.use();
        glActiveTexture(GL_TEXTURE1);
        tex2.use();
        glBindVertexArray(VAO);

        myShaiders.use();

        mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glUniformMatrix4fv(glGetUniformLocation(myShaiders.getId(), "view"), 
                            1, 
                            GL_FALSE, 
                            (float*)&view);
        mat4 proj = perspective(radians(45.0f), float(wiewWidth)/wiewHeight, 0.001f, 100.0f);
        glUniformMatrix4fv(glGetUniformLocation(myShaiders.getId(), "perspective"), 
                            1, 
                            GL_FALSE, 
                            (float*)&proj);
        grad = int(grad + 1)%360;
        auto axisW = vec3(1, 0, 0);
        auto axisS = vec3(0, 1, 0);
        rot = rotate(mat4(1), radians(float(i%360)), axisW) * rotate(mat4(1), radians(float(i%360)), axisS);

       int maxObj = stoi(argv[5]);
        for(int j = 0; j < sqrt(maxObj); j++){
            for(int k = 0; k < sqrt(maxObj); k++){
                auto trans = translate(mat4(1.0f), vec3(j/sqrt(maxObj),k/sqrt(maxObj),0))*rot;
                glUniformMatrix4fv(glGetUniformLocation(myShaiders.getId(), "world"), 
                                    1, 
                                    GL_FALSE, 
                                    (float*)&trans);
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
