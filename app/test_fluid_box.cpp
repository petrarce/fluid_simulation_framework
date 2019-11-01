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
using namespace CompactNSearch;


int main(int argc, char** argv)
{
	std::cout << "Welcome to the learnSPH framework!!" << std::endl;
	std::cout << "Generating a sample scene...";

	assert(argc == 9);
	Vector3R upper_corner = {stod(argv[1]), stod(argv[2]), stod(argv[3])};
	Vector3R lover_corner = {stod(argv[4]), stod(argv[5]), stod(argv[6])};
	Real sampling_distance = stod(argv[7]);
	double compactSupportFactor = stod(argv[8]);
	NeighborhoodSearch ns(sampling_distance*compactSupportFactor);


	NormalPartDataSet* fluidParticles = 
		static_cast<NormalPartDataSet*>(learnSPH::ParticleSampler::sample_normal_particles(upper_corner, 
															lover_corner, 
															1000, 
															sampling_distance));

	auto fluidPartilesPset = ns.add_point_set((Real*)(fluidParticles->getParticlePositions().data()),
						fluidParticles->getNumberOfParticles(),
						true,
						true, 
						true);
	std::cout<< "number of fluid particles: " << fluidParticles->getNumberOfParticles() << endl;

	vector<Vector3R> dummyVector;
	BorderPartDataSet dummyBorderParticles(dummyVector, 1, 1);
	ns.add_point_set((Real*)dummyBorderParticles.getParticlePositions().data(), 
						dummyBorderParticles.getNumberOfParticles(),
						false, true, true);

	ns.update_point_sets();

	vector<vector<vector<unsigned int>>> particleNeighbors;
	particleNeighbors.resize(fluidParticles->getNumberOfParticles());
	for(int i = 0; i < fluidParticles->getNumberOfParticles(); i++){
		ns.find_neighbors(fluidPartilesPset, i, particleNeighbors[i]);
	}

	learnSPH::Solver::calculate_dencities(*fluidParticles, 
											dummyBorderParticles, 
											particleNeighbors,
											sampling_distance*compactSupportFactor);
	// Generate particles
	std::string filename = "../res/fluid_particle_data_set.vtk";
	learnSPH::saveParticlesToVTK(filename, 
									fluidParticles->getParticlePositions(), 
									fluidParticles->getParticleDencities(), 
									fluidParticles->getParticleVelocities());

	for(Real density : fluidParticles->getParticleDencities()){
		fprintf(stderr, "%f\n", density);
	}
 
	delete fluidParticles;
	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res`. You can visualize them with Paraview." << std::endl;


	return 0;
}