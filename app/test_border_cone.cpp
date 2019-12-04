#include <stdlib.h>     // rand
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // std::max

#include <Eigen/Dense>
#include <storage.h>
#include <types.hpp>
#include <particle_sampler.h>
#include <CompactNSearch>

#include <vtk_writer.h>
#include <solver.h>
#include <chrono>

using namespace learnSPH;

int main(int argc, char** argv)
{
	std::cout << "Welcome to the learnSPH framework!!" << std::endl;
	std::cout << "Generating a sample scene..." << std::endl;

	assert(argc == 11);

	Vector3R lowerConeCenter = Vector3R(stod(argv[1]), stod(argv[2]), stod(argv[3]));
	Vector3R upperConeCenter = Vector3R(stod(argv[4]), stod(argv[5]), stod(argv[6]));
	Real lowerConeRad = stod(argv[7]);
	Real upperConeRad = stod(argv[8]);
	Real samplingDistance = stod(argv[9]);
	Real eta = stod(argv[10]);


	BorderSystem* borderConeParticles =
			sample_border_cone(
									lowerConeRad,
									lowerConeCenter,
									upperConeRad,
									upperConeCenter,
									1000,
									samplingDistance,
									eta);
	
	BorderSystem* borderCubeParticles =
			sample_border_box(
									lowerConeCenter,
									upperConeCenter,
									1000,
									samplingDistance,
									eta,
									true);


	std::string filename = "../res/border_cone.vtk";
	vector<Vector3R> dummyVector(borderConeParticles->size());
	learnSPH::saveParticlesToVTK(filename, 
									borderConeParticles->getPositions(), 
									borderConeParticles->getVolumes(), 
									dummyVector);
 
 	filename = "../res/border_cube.vtk";
	dummyVector.resize(borderCubeParticles->size());
	learnSPH::saveParticlesToVTK(filename, 
									borderCubeParticles->getPositions(), 
									borderCubeParticles->getVolumes(), 
									dummyVector);

	delete borderConeParticles;
	delete borderCubeParticles;

	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res`. You can visualize them with Paraview." << std::endl;

	return 0;
}