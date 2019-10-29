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

	assert(argc == 17);
	Vector3R upper_corner = {stod(argv[1]), stod(argv[2]), stod(argv[3])};
	Vector3R lover_corner = {stod(argv[4]), stod(argv[5]), stod(argv[6])};
	Real sampling_distance = stod(argv[7]);



	NormalPartDataSet* particles = 
		static_cast<NormalPartDataSet*>(learnSPH::ParticleSampler::sample_normal_particles(upper_corner, 
															lover_corner, 
															0.01, 
															sampling_distance));



	// Generate particles
	std::string filename = "../res/fluid_particle_data_set.vtk";
	learnSPH::saveParticlesToVTK(filename, 
									particles->getParticlePositions(), 
									particles->getParticleDencities(), 
									particles->getParticleVelocities());

 
	Vector3R tp1 = {stod(argv[8]), stod(argv[9]),stod(argv[10])};
	Vector3R tp2 = {stod(argv[11]), stod(argv[12]),stod(argv[13])};
	Vector3R tp3 = {stod(argv[14]), stod(argv[15]),stod(argv[16])};

	BorderPartDataSet* brdParticles = 
		static_cast<BorderPartDataSet*>(learnSPH::ParticleSampler::sample_border_particles(
			tp1, tp2, tp3, 0.01, sampling_distance));

	vector<Real> dummyVec1;
	dummyVec1.resize(brdParticles->getNumberOfParticles());
	vector<Vector3R> dummyVec2;
	dummyVec2.resize(brdParticles->getNumberOfParticles());

	filename = "../res/border_particle_data_set.vtk";
	learnSPH::saveParticlesToVTK(filename, 
									brdParticles->getParticlePositions(),
									dummyVec1, 
									dummyVec2);

 	delete particles;
 	delete brdParticles;
	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res`. You can visualize them with Paraview." << std::endl;


	return 0;
}