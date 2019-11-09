#include <stdlib.h>     // rand
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // std::max

#include <Eigen/Dense>
#include <data_set.h>
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

	assert(argc == 9);

	Vector3R lower_corner = Vector3R(stod(argv[1]), stod(argv[2]), stod(argv[3]));
	Vector3R upper_corner = Vector3R(stod(argv[4]), stod(argv[5]), stod(argv[6]));

	Real sampling_distance = stod(argv[7]);
	bool hexagonal = (bool)stod(argv[8]);

	BorderPartDataSet* borderParticles = static_cast<BorderPartDataSet*>(ParticleSampler::sample_border_box(lower_corner, upper_corner, 1000, sampling_distance, hexagonal));

	std::string filename = "../res/border_particle_data_set_with_border.vtk";
	vector<Vector3R> dummyVector(borderParticles->getNumberOfParticles());
	learnSPH::saveParticlesToVTK(filename, 
									borderParticles->getParticlePositions(), 
									borderParticles->getParticleVolume(), 
									dummyVector);
 
	delete borderParticles;

	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res`. You can visualize them with Paraview." << std::endl;

	return 0;
}