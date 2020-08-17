#include <stdlib.h>     // rand
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // std::max

#include <Eigen/Dense>
#include <learnSPH/core/storage.h>
#include <types.hpp>
#include <learnSPH/core/particle_sampler.h>
#include <CompactNSearch>

#include <learnSPH/core/vtk_writer.h>
#include <learnSPH/simulation/solver.h>
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

	BorderSystem* borderConeParticles = sample_border_cone(lowerConeRad, lowerConeCenter, upperConeRad, upperConeCenter, 1000, samplingDistance, eta);

	std::string filename = "res/border_cone.vtk";
	vector<Vector3R> dummyVector(borderConeParticles->size());
	learnSPH::saveParticlesToVTK(filename, borderConeParticles->getPositions(), borderConeParticles->getVolumes(), dummyVector);

	vector<Vector3R> borderParticles;
	borderParticles.clear();
	learnSPH::sample_sphere(borderParticles, (lowerConeCenter - upperConeCenter).norm(), lowerConeCenter + upperConeCenter / 2, samplingDistance);

	BorderSystem sphere(borderParticles, 1000, samplingDistance, eta);
	filename = "res/border_sphere.vtk";
	vector<Real> dummyVectorReal(borderParticles.size());
	learnSPH::saveParticlesToVTK(filename, sphere.getPositions(), sphere.getVolumes(), borderParticles);

	delete borderConeParticles;

	std::cout << "completed!" << std::endl;

	return 0;
}