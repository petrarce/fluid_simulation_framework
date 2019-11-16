#include <stdlib.h>     // rand
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // std::max

#include <Eigen/Dense>
#include <types.hpp>
#include <marching_cubes.h>

#include <vtk_writer.h>
using namespace learnSPH;


int main(int argc, char** argv)
{
	std::cout << "Welcome to the learnSPH framework!!" << std::endl;
	std::cout << "Generating a sample scene...";

	assert(argc == 14);
	Vector3R lower_corner = {stod(argv[1]), stod(argv[2]), stod(argv[3])};
	Vector3R upper_corner = {stod(argv[4]), stod(argv[5]), stod(argv[6])};
	Vector3R cubeResolution = {stod(argv[7]), stod(argv[8]), stod(argv[9])};
	Vector3R sphereCenter = {stod(argv[10]), stod(argv[11]), stod(argv[12])};
	Real sphereRadius = stod(argv[13]);

	vector<Vector3R> triangle_mesh;
	Sphere sphr(sphereRadius, sphereCenter);
	MarchingCubes mcb;
	mcb.init(lower_corner, upper_corner, cubeResolution);
	mcb.setObject(&sphr);
	mcb.getTriangleMesh(triangle_mesh);

	// Generate particles
	vector<Real> dummyScalarVec;
	dummyScalarVec.resize(triangle_mesh.size());
	
	std::string filename = "../res/marching_cube_sphere_mesh.vtk";
	learnSPH::saveParticlesToVTK(filename, 
									triangle_mesh, 
									dummyScalarVec, 
									triangle_mesh);

	Thorus thr(sphereRadius, 0.5*sphereRadius, sphereCenter);
	mcb.setObject(&thr);
	mcb.getTriangleMesh(triangle_mesh);
	dummyScalarVec.resize(triangle_mesh.size());

	filename = "../res/marching_cube_thorus_mesh.vtk";
	learnSPH::saveParticlesToVTK(filename, 
									triangle_mesh, 
									dummyScalarVec, 
									triangle_mesh);

 
	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res`. You can visualize them with Paraview." << std::endl;


	return 0;
}