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
	int genWithBorder = stoi(argv[8]);
	NeighborhoodSearch ns(sampling_distance*compactSupportFactor);


	//generate fluid particle cube
	auto t0 = std::chrono::high_resolution_clock::now();
	NormalPartDataSet* fluidParticles = 
		static_cast<NormalPartDataSet*>(learnSPH::ParticleSampler::sample_normal_particles(upper_corner, 
															lover_corner, 
															1000, 
															sampling_distance));
	auto t1 = std::chrono::high_resolution_clock::now();
	printf("fluid particle sampling time: %ld ms\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

	auto fluidPartilesPset = ns.add_point_set((Real*)(fluidParticles->getParticlePositions().data()),
						fluidParticles->getNumberOfParticles(),
						true,
						true, 
						true);
	std::cout<< "number of fluid particles: " << fluidParticles->getNumberOfParticles() << endl;

	//generate border particle box if specified
	upper_corner = upper_corner+
					(upper_corner - lover_corner).normalized()*sampling_distance;
	lover_corner = lover_corner + 
					(lover_corner - upper_corner).normalized()*sampling_distance;
	t0 = std::chrono::high_resolution_clock::now();
	BorderPartDataSet* borderParticles = 
		static_cast<BorderPartDataSet*>(learnSPH::ParticleSampler::sample_border_box(upper_corner, 
															lover_corner, 
															1000, 
															sampling_distance));
	t1 = std::chrono::high_resolution_clock::now();
	printf("border particle sampling time: %ld ms\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

	auto borderParticlesPset = ns.add_point_set((Real*)borderParticles->getParticlePositionsData(),
						borderParticles->getNumberOfParticles(),
						false,
						true, 
						true);
	std::cout<< "number of border particles: " << borderParticles->getNumberOfParticles() << endl;


	ns.update_point_sets();

	vector<vector<vector<unsigned int>>> particleNeighbors;
	particleNeighbors.resize(fluidParticles->getNumberOfParticles());
	for(int i = 0; i < fluidParticles->getNumberOfParticles(); i++){
		ns.find_neighbors(fluidPartilesPset, i, particleNeighbors[i]);
	}

	t0 = std::chrono::high_resolution_clock::now();
	learnSPH::Solver::calculate_dencities(*fluidParticles, *borderParticles, particleNeighbors);
	t1 = std::chrono::high_resolution_clock::now();
	printf("density calcultaions time: %ld ms\n", 
		std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
	// Generate particles
	std::string filename = "../res/fluid_particle_data_set.vtk";
	learnSPH::saveParticlesToVTK(filename, 
									fluidParticles->getParticlePositions(), 
									fluidParticles->getParticleDencities(), 
									fluidParticles->getParticleVelocities());

	filename = "../res/boreder_particle_data_set.vtk";
	learnSPH::saveParticlesToVTK(filename, 
									borderParticles->getParticlePositions(), 
									borderParticles->getParticleVolume(), 
									borderParticles->getParticlePositions());

 
	delete fluidParticles;
	delete borderParticles;
	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res`. You can visualize them with Paraview." << std::endl;


	return 0;
}