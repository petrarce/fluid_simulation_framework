#include <stdlib.h>     // rand
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // std::max

#include <Eigen/Dense>
#include <data_set.h>
#include <types.hpp>
#include <particle_sampler.h>

#include <vtk_writer.h>

int main(int argc, char** argv)
{
	std::cout << "Welcome to the learnSPH framework!!" << std::endl;
	std::cout << "Generating a sample scene...";

	assert(argc == 7+1);
	Vector3R upper_corner = {stod(argv[1]), stod(argv[2]), stod(argv[3])};
	Vector3R lover_corner = {stod(argv[4]), stod(argv[5]), stod(argv[6])};
	Real sampling_distance = stod(argv[7]);

	NormalPartDataSet* particles = 
		static_cast<NormalPartDataSet*>(learnSPH::ParticleSampler::sample_normal_particles(upper_corner, 
															lover_corner, 
															0.01, 
															sampling_distance));



	// Generate particles
	std::string filename = "../res/sample_particle_data_set.vtk";
	learnSPH::saveParticlesToVTK(filename, 
									particles->getParticlePositions(), 
									particles->getParticleDencities(), 
									particles->getParticleVelocities());

 	delete particles;
	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res`. You can visualize them with Paraview." << std::endl;


	return 0;
}