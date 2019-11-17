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
	assert(argc == 22);
	std::cout << "Welcome to the learnSPH framework!!" << std::endl;
	std::cout << "Generating test sample for Assignment 2...";

	Vector3R upper_corner_fluid = {stod(argv[1]),stod(argv[2]),stod(argv[3])};
	Vector3R lower_corner_fluid = {stod(argv[4]),stod(argv[5]),stod(argv[6])};
	Vector3R upper_corner_box = {stod(argv[7]),stod(argv[8]),stod(argv[9])};
	Vector3R lover_corner_box = {stod(argv[10]),stod(argv[11]),stod(argv[12])};
	Real sampling_distance = stod(argv[13]);
	Real eta = stod(argv[14]);
	Real pressureStiffness = stod(argv[15]);
	Real viscosity = stod(argv[16]);
	Real friction = stod(argv[17]);
	bool with_smoothing = stoi(argv[18]);
	Real render_step = (stod(argv[19])); // time frame
	Real simulateDuration = (stod(argv[20])); // duration of the simulation
	string expName = argv[21]; // name of experience

	NormalPartDataSet* fluidParticles = sample_fluid_cube(upper_corner_fluid, lower_corner_fluid, 1000, sampling_distance, eta);

	cout << "Number of fluid particles: " << fluidParticles->size() << endl;

	BorderPartDataSet* borderParticles = sample_border_box(lover_corner_box, upper_corner_box, 3000, sampling_distance * 0.5, eta * 0.5, true);

	cout << "Number of border particles: " << borderParticles->size() << endl;

	NeighborhoodSearch ns(fluidParticles->getCompactSupport());

	ns.add_point_set((Real*)(fluidParticles->getPositions().data()), fluidParticles->size(), true);

	ns.add_point_set((Real*)borderParticles->getPositions().data(), borderParticles->size(), false);

	vector<vector<vector<unsigned int>>> neighbors;

	neighbors.resize(fluidParticles->size());

	const Vector3R gravity(0.0, -9.7, 0.0);

	vector<Vector3R>& particleForces = fluidParticles->getExternalForces();

	for(unsigned int i = 0; i < particleForces.size(); i++) particleForces[i] = fluidParticles->getMass() * gravity;

	unsigned int nsamples = int(simulateDuration / render_step);

	cout << "Duration: " << simulateDuration << endl;
	cout << "Default time step: "<< render_step << endl;
	cout << "number of frames: "<< simulateDuration / render_step << endl;

	string filename = "res/assignment2/border.vtk";

	vector<Vector3R> dummyVector(borderParticles->size());

	learnSPH::saveParticlesToVTK(filename, borderParticles->getPositions(), borderParticles->getVolumes(), dummyVector);

	for (unsigned int t = 0; t < nsamples; t++) {

		Real timeSimulation = 0;

		int physical_steps = 0;

		while (timeSimulation < 1) {
			ns.update_point_sets();

			for(int i = 0; i < fluidParticles->size(); i++) ns.find_neighbors(0, i, neighbors[i]);

			learnSPH::calculate_dencities(fluidParticles, borderParticles, neighbors, fluidParticles->getSmoothingLength());


			vector<Vector3R> accelerations(fluidParticles->size(), gravity);

			learnSPH::calculate_acceleration(
											accelerations,
											fluidParticles,
											borderParticles,
											neighbors,
											viscosity,
											friction,
											pressureStiffness,
											fluidParticles->getSmoothingLength());

			Real velocityCap = 100.0;
			Real vMaxNorm = 0.0;
			auto fluidVelocities = fluidParticles->getVelocities().data();

			for (int iVelo = 0; iVelo < fluidParticles->size(); iVelo++) vMaxNorm = max(fluidVelocities[iVelo].norm(), vMaxNorm);

			vMaxNorm = min(vMaxNorm, velocityCap);

			Real logic_step_upper_bound = 0.5 * (fluidParticles->getDiameter() / vMaxNorm);
			Real logic_time_step;

			if (timeSimulation * render_step + logic_step_upper_bound >= render_step){
				logic_time_step = (1 - timeSimulation) * render_step;
				timeSimulation = 1;
			} else {
				logic_time_step = logic_step_upper_bound;
				timeSimulation += logic_time_step / render_step;
			}

			if (!with_smoothing)
				learnSPH::symplectic_euler(accelerations, fluidParticles, logic_time_step);
			else
				learnSPH::smooth_symplectic_euler(accelerations, fluidParticles, neighbors, 0.5, logic_time_step, fluidParticles->getSmoothingLength());

			physical_steps++;
		}
		cout << "[" << physical_steps << "] physical updates were carried out for rendering frame [" << t << "]" << endl;

		string filename = "res/assignment2/" + expName + '_' + std::to_string(t) + ".vtk";

		learnSPH::saveParticlesToVTK(filename, fluidParticles->getPositions(), fluidParticles->getDensities(), fluidParticles->getVelocities());
	}
	delete fluidParticles;
	std::cout << "completed!" << std::endl;
	std::cout << "The scene files have been saved in the folder `<build_folder>/res/assignment2/`. You can visualize them with Paraview." << std::endl;

	return 0;
}
