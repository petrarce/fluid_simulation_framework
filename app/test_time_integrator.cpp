#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>

#include <Eigen/Dense>
#include <data_set.h>
#include <types.hpp>
#include <particle_sampler.h>
#include <CompactNSearch>

#include <vtk_writer.h>
#include <solver.h>
#include <chrono>

using namespace CompactNSearch;
using namespace learnSPH;

int main(int argc, char** argv)
{
	assert(argc == 23);
	std::cout << "Welcome to the learnSPH framework!!" << std::endl;
	std::cout << "Generating test sample for Assignment 2...";

	Vector3R upper_corner_fluid = {stod(argv[1]),stod(argv[2]),stod(argv[3])};
	Vector3R lower_corner_fluid = {stod(argv[4]),stod(argv[5]),stod(argv[6])};
	Vector3R upper_corner_box = {stod(argv[7]),stod(argv[8]),stod(argv[9])};
	Vector3R lover_corner_box = {stod(argv[10]),stod(argv[11]),stod(argv[12])};
	Real sampling_distance = stod(argv[13]);
	Real compactSupportFactor = stod(argv[14]);
	Real preasureStiffness = stod(argv[15]);
	Real viscosity = stod(argv[16]);
	Real friction = stod(argv[17]);
	bool with_smoothing = stoi(argv[18]);
	bool withNavierStokes = stoi(argv[19]);
	Real render_step = (stod(argv[20])); // time frame
	Real simulateDuration = (stod(argv[21])); // duration of the simulation
	string expName = argv[22]; // name of experience

	NormalPartDataSet* fluidParticles = sample_fluid_cube(upper_corner_fluid, lower_corner_fluid, 1000, sampling_distance);

	fluidParticles->setCompactSupportFactor(compactSupportFactor);

	NeighborhoodSearch ns(fluidParticles->getCompactSupport());

	auto fluidPointSet = ns.add_point_set((Real*)(fluidParticles->getParticlePositions().data()), fluidParticles->getNumberOfParticles(), true);

	cout << "Number of fluid particles: " << fluidParticles->getNumberOfParticles() << endl;

	BorderPartDataSet* borderParticles = sample_border_box(lover_corner_box, upper_corner_box, 3000, sampling_distance * 0.5, true);

	cout << "Number of border particles: " << borderParticles->getNumberOfParticles() << endl;

	ns.add_point_set((Real*)borderParticles->getParticlePositions().data(), borderParticles->getNumberOfParticles(), false);

	vector<vector<vector<unsigned int>>> particleNeighbors;

	particleNeighbors.resize(fluidParticles->getNumberOfParticles());

	const Vector3R gravity(0.0, -9.7, 0.0);

	vector<Vector3R>& particleForces = fluidParticles->getExternalForces();

	for(unsigned int i = 0; i < particleForces.size(); i++) particleForces[i] = fluidParticles->getParticleMass() * gravity;

	unsigned int nsamples = int(simulateDuration / render_step);

	cout << "Duration: " << simulateDuration << endl;
	cout << "Default time step: "<< render_step << endl;
	cout << "number of frames: "<< simulateDuration / render_step << endl;

	string filename = "res/assignment2/border.vtk";

	vector<Vector3R> dummyVector(borderParticles->getNumberOfParticles());

	learnSPH::saveParticlesToVTK(filename, borderParticles->getParticlePositions(), borderParticles->getParticleVolume(), dummyVector);

	for (unsigned int t = 0; t < nsamples; t++) {

		Real timeSimulation = 0;

		int physical_steps = 0;

		while (timeSimulation < 1) {
			ns.update_point_sets();

			for(int i = 0; i < fluidParticles->getNumberOfParticles(); i++) ns.find_neighbors(fluidPointSet, i, particleNeighbors[i]);

			learnSPH::calculate_dencities(fluidParticles, borderParticles, particleNeighbors, fluidParticles->getSmoothingLength());


			vector<Vector3R> fluidParticlesAccelerations(fluidParticles->getNumberOfParticles(), gravity);

			learnSPH::calculate_acceleration(
											fluidParticlesAccelerations,
											fluidParticles,
											borderParticles,
											particleNeighbors,
											viscosity,
											friction,
											preasureStiffness,
											fluidParticles->getSmoothingLength());

			Real velocityCap = 300.0;
			Real vMaxNorm = 0;
			auto fluidParticlesVelocity = fluidParticles->getParticleVelocities().data();

			for (int iVelo = 0; iVelo < fluidParticles->getNumberOfParticles(); iVelo++) if(fluidParticlesVelocity[iVelo].norm() > vMaxNorm) vMaxNorm = (fluidParticlesVelocity[iVelo]).norm();

			vMaxNorm = min(vMaxNorm, velocityCap);

			Real logic_step_upper_bound = 0.5 * (fluidParticles->getParticleDiameter() / vMaxNorm);
			Real logic_time_step;

			if (timeSimulation * render_step + logic_step_upper_bound >= render_step){
				logic_time_step = (1 - timeSimulation) * render_step;
				timeSimulation = 1;
			} else {
				logic_time_step = logic_step_upper_bound;
				timeSimulation += logic_time_step / render_step;
			}

			if (!with_smoothing) {
				learnSPH::symplectic_euler(fluidParticlesAccelerations, fluidParticles, logic_time_step);
			} else {
				learnSPH::smooth_symplectic_euler(
												fluidParticlesAccelerations,
												fluidParticles,
												particleNeighbors,
												0.5,
												logic_time_step,
												fluidParticles->getSmoothingLength());
			}
			physical_steps++;
		}
		cout << "[" << physical_steps << "] physical updates were carried out for rendering frame [" << t << "]" << endl;

		string filename = "res/assignment2/" + expName + '_' + std::to_string(t) + ".vtk";

		learnSPH::saveParticlesToVTK(filename, fluidParticles->getParticlePositions(), fluidParticles->getParticleDencities(), fluidParticles->getParticleVelocities());
	}
	delete fluidParticles;
	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res/assignment2/`. You can visualize them with Paraview." << std::endl;

	return 0;
}
