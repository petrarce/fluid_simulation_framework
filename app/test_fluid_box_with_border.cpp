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

Vector3d crossPoint(Vector3d& p1, Vector3d& v1,
							Vector3d& p2, Vector3d& v2)
{
	//abort lines are on the different planes
	assert(fabs((v1.cross(v2)).dot(p2-p1)) < 10e-6);
	//abort if vectors are colinear (infinity points of intersections or parallel)
	assert(v1.cross(v2).norm() > 10e-6);

	Matrix2d A;
	Matrix<double, 2,3> projMatr;
	projMatr.row(0) = v1.normalized();
	projMatr.row(1) = ((v1.cross(v2)).cross(v1)).normalized();
	assert(projMatr.row(0).dot(projMatr.row(1)) < 10e-6);

	cout << projMatr << endl;

	Vector2d p1n = projMatr * p1;
	Vector2d v1n = projMatr * v1;
	Vector2d p2n = projMatr * p2;
	Vector2d v2n = projMatr * v2;


	A.row(0) =  Vector2d(v1n[1], -v1n[0]);
	A.row(1) =  Vector2d(v2n[1], -v2n[0]);

	Vector2d b;
	b = {p1n[0]*v1n[1]-p1n[1]*v1n[0], 
		 p2n[0]*v2n[1]-p2n[1]*v2n[0]};
	Vector3d sol3d = projMatr.transpose() * A.colPivHouseholderQr().solve(b);

	return sol3d;
}


int main(int argc, char** argv)
{
	std::cout << "Welcome to the learnSPH framework!!" << std::endl;
	std::cout << "Generating a sample scene...";

	assert(argc == 15);
	Vector3R upper_corner = {stod(argv[1]), stod(argv[2]), stod(argv[3])};
	Vector3R lower_corner = {stod(argv[4]), stod(argv[5]), stod(argv[6])};
	Vector3R upper_corner_box = {stod(argv[7]), stod(argv[8]), stod(argv[9])};
	Vector3R lower_corner_box = {stod(argv[10]), stod(argv[11]), stod(argv[12])};

	Real sampling_distance = stod(argv[13]);
	double compactSupportFactor = stod(argv[14]);


	NormalPartDataSet* fluidParticles = 
		static_cast<NormalPartDataSet*>(learnSPH::ParticleSampler::sample_normal_particles(upper_corner, 
															lower_corner, 
															1000, 
															sampling_distance));
	fluidParticles->setCompactSupportFactor(compactSupportFactor);
	NeighborhoodSearch ns(fluidParticles->getCompactSupport());
	auto fluidPartilesPset = ns.add_point_set((Real*)(fluidParticles->getParticlePositions().data()),
						fluidParticles->getNumberOfParticles(),
						true,
						true, 
						true);
	std::cout<< "number of fluid particles: " << fluidParticles->getNumberOfParticles() << endl;


	Real offset = fluidParticles->getCompactSupport() + 1;
	BorderPartDataSet* borderParticles = 
		static_cast<BorderPartDataSet*>(learnSPH::ParticleSampler::sample_border_box(
			upper_corner_box, 
			lower_corner_box, 
			3000, 0.5*sampling_distance, true));
	

	ns.add_point_set((Real*)borderParticles->getParticlePositions().data(), 
						borderParticles->getNumberOfParticles(),
						false, true, true);

	ns.update_point_sets();

	vector<vector<vector<unsigned int>>> particleNeighbors;
	particleNeighbors.resize(fluidParticles->getNumberOfParticles());
	for(int i = 0; i < fluidParticles->getNumberOfParticles(); i++){
		ns.find_neighbors(fluidPartilesPset, i, particleNeighbors[i]);
	}

	learnSPH::Solver::calculate_dencities(*fluidParticles, 
											*borderParticles, 
											particleNeighbors,
											fluidParticles->getSmoothingLength());
	// Generate particles
	std::string filename = "../res/fluid_particle_data_set_with_border.vtk";
	learnSPH::saveParticlesToVTK(filename, 
									fluidParticles->getParticlePositions(), 
									fluidParticles->getParticleDencities(), 
									fluidParticles->getParticleVelocities());

	filename = "../res/border_particle_data_set_with_border.vtk";
	vector<Vector3R> dummyVector(borderParticles->getNumberOfParticles());
	learnSPH::saveParticlesToVTK(filename, 
									borderParticles->getParticlePositions(), 
									borderParticles->getParticleVolume(), 
									dummyVector);

	for(Real density : fluidParticles->getParticleDencities()){
		fprintf(stderr, "%f\n", density);
	}
 
	delete fluidParticles;
	delete borderParticles;
	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res`. You can visualize them with Paraview." << std::endl;


	return 0;
}