#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <Eigen/Dense>
#include <types.hpp>
#include <learnSPH/surf_reconstr/marching_cubes_deprecated.h>

#include <learnSPH/core/vtk_writer.h>
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

	MarchingCubes mcb(lower_corner, upper_corner, cubeResolution);

	Sphere sphr(sphereRadius, sphereCenter, lower_corner, upper_corner, cubeResolution);
	Thorus thr(sphereRadius, 0.5 * sphereRadius, sphereCenter, lower_corner, upper_corner, cubeResolution);

	vector<Vector3R> triangle_mesh;
	mcb.setObject(&sphr);
	mcb.getTriangleMesh(triangle_mesh);

	vector<array<size_t, 3>> triangles;
	for(size_t i = 0; i < triangle_mesh.size(); i += 3) triangles.push_back({i, i + 1, i + 2});

	std::string filename = "res/marching_cube_sphere_mesh.vtk";
	learnSPH::saveTriMeshToVTK(filename, triangle_mesh, triangles);

	triangle_mesh.clear();
	mcb.setObject(&thr);
	mcb.getTriangleMesh(triangle_mesh);

	triangles.clear();
	for(size_t i = 0; i < triangle_mesh.size(); i += 3) triangles.push_back({i, i + 1, i + 2});

	filename = "res/marching_cube_thorus_mesh.vtk";
	learnSPH::saveTriMeshToVTK(filename, triangle_mesh, triangles);

	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res`. You can visualize them with Paraview." << std::endl;

	return 0;
}
