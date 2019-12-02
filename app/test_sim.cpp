#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>

#include <Eigen/Dense>
#include <storage.h>
#include <types.hpp>
#include <particle_sampler.h>
#include <CompactNSearch>

#include <vtk_writer.h>
#include <solver.h>
#include <chrono>

#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

using namespace CompactNSearch;
using namespace learnSPH;


void save_vectors(const std::string &path, std::vector<Vector3R> &data)
{
	std::ofstream os(path, std::ios::binary);

	cereal::BinaryOutputArchive boa(os);

	boa(data.size());

	for (Vector3R &vec : data) boa(vec(0), vec(1), vec(2));
}

void save_scalars(const std::string &path, std::vector<Real> &data)
{
	std::ofstream os(path, std::ios::binary);

	cereal::BinaryOutputArchive boa(os);

	boa(data.size());

	for (auto val : data) boa(val);
}

int main(int argc, char** argv)
{
	assert(argc == 22);

	std::cout << "Simulation running" << std::endl;

	Vector3R lower_corner_fluid(stod(argv[1]), stod(argv[2]), stod(argv[3]));
	Vector3R upper_corner_fluid(stod(argv[4]), stod(argv[5]), stod(argv[6]));

	Vector3R lower_corner_box(stod(argv[7]), stod(argv[8]), stod(argv[9]));
	Vector3R upper_corner_box(stod(argv[10]), stod(argv[11]), stod(argv[12]));

	auto box_center = (lower_corner_box + upper_corner_box) / 2.0;
	auto max_shift = (box_center - lower_corner_box).norm() * 1.2;

	Real sampling_distance = stod(argv[13]);
	Real eta = stod(argv[14]);
	Real stiffness = stod(argv[15]);
	Real viscosity = stod(argv[16]);
	Real friction = stod(argv[17]);
	bool do_velo_smooth = stoi(argv[18]);

	Real render_step = stod(argv[19]);
	Real sim_duration = stod(argv[20]);
	string sim_name = argv[21];

	FluidSystem* fluidParticles = sample_fluid_cube(lower_corner_fluid, upper_corner_fluid, 1000.0, sampling_distance, eta);

	cout << "Number of fluid particles: " << fluidParticles->size() << endl;

	BorderSystem* borderParticles = sample_border_box(lower_corner_box, upper_corner_box, 3000.0, sampling_distance * 0.5, eta * 0.5, true);

	cout << "Number of border particles: " << borderParticles->size() << endl;

	NeighborhoodSearch ns(fluidParticles->getCompactSupport());

	ns.add_point_set((Real*)(fluidParticles->getPositions().data()), fluidParticles->size(), true);

	ns.add_point_set((Real*)borderParticles->getPositions().data(), borderParticles->size(), false);

	const Vector3R gravity(0.0, -9.7, 0.0);

	vector<Vector3R>& particleForces = fluidParticles->getExternalForces();

	for(unsigned int i = 0; i < particleForces.size(); i++) particleForces[i] = fluidParticles->getMass() * gravity;

	cout << "Diameter: " << fluidParticles->getDiameter() << endl;
	cout << "Duration: " << sim_duration << endl;
	cout << "Default time step: "<< render_step << endl;
	cout << "number of frames: "<< sim_duration / render_step << endl;

	string filename = "res/assignment3/border.vtk";

	vector<Vector3R> dummyVector(borderParticles->size());

	learnSPH::saveParticlesToVTK(filename, borderParticles->getPositions(), borderParticles->getVolumes(), dummyVector);

	unsigned int nsamples = int(sim_duration / render_step);

	for (unsigned int t = 0; t < nsamples; t++) {

		Real timeSimulation = 0;

		int physical_steps = 0;

		while (timeSimulation < 1) {
			fluidParticles->findNeighbors(ns);

			learnSPH::calculate_dencities(fluidParticles, borderParticles, fluidParticles->getNeighbors(), fluidParticles->getSmoothingLength());


			vector<Vector3R> accelerations(fluidParticles->size(), gravity);

			learnSPH::calculate_acceleration(
											accelerations,
											fluidParticles,
											borderParticles,
											fluidParticles->getNeighbors(),
											viscosity,
											friction,
											stiffness,
											fluidParticles->getSmoothingLength());

			Real vMaxNorm = 0.0;

			vector<bool> &fluidActiveness = fluidParticles->getActiveness();

			vector<Vector3R> &fluidVelocities = fluidParticles->getVelocities();

			for (int i = 0; i < fluidParticles->size(); i++) if (fluidActiveness[i]) vMaxNorm = max(fluidVelocities[i].norm(), vMaxNorm);

			Real logic_step_upper_bound = 0.5 * (fluidParticles->getDiameter() / vMaxNorm);

			Real logic_time_step;

			if (timeSimulation * render_step + logic_step_upper_bound >= render_step) {
				logic_time_step = (1 - timeSimulation) * render_step;
				timeSimulation = 1;
			} else {
				logic_time_step = logic_step_upper_bound;
				timeSimulation += logic_time_step / render_step;
			}

			if (!do_velo_smooth)
				learnSPH::symplectic_euler(accelerations, fluidParticles, logic_time_step);
			else
				learnSPH::smooth_symplectic_euler(accelerations, fluidParticles, fluidParticles->getNeighbors(), 0.5, logic_time_step, fluidParticles->getSmoothingLength());

			Real velocityCap = 50.0;

			vector<Vector3R> &fluidPositions = fluidParticles->getPositions();

			for (int i = 0; i < fluidParticles->size(); i++) {

				if (!fluidActiveness[i])
					continue;

				if (fluidVelocities[i].norm() > velocityCap)
					fluidVelocities[i] = velocityCap * fluidVelocities[i].normalized();

				if ((fluidPositions[i] - box_center).norm() > max_shift)
					fluidActiveness[i] = false;
			}
			physical_steps++;
		}
		cout << "\n[" << physical_steps << "] physical updates were carried out for rendering frame [" << t << "]" << endl;

		string filename = "res/assignment3/" + sim_name + '_' + std::to_string(t) + ".vtk";

		learnSPH::saveParticlesToVTK(filename, fluidParticles->getPositions(), fluidParticles->getDensities(), fluidParticles->getVelocities());

		vector<Real> params;

		params.push_back(fluidParticles->getCompactSupport());
		params.push_back(fluidParticles->getSmoothingLength());
		params.push_back(fluidParticles->getMass());
		params.push_back(fluidParticles->getRestDensity());

		filename = "res/assignment3/" + sim_name + "_params_" + std::to_string(t) + ".cereal";

		save_scalars(filename, params);

		filename = "res/assignment3/" + sim_name + "_positions_" + std::to_string(t) + ".cereal";

		save_vectors(filename, fluidParticles->getPositions());

		filename = "res/assignment3/" + sim_name + "_densities_" + std::to_string(t) + ".cereal";

		save_scalars(filename, fluidParticles->getDensities());
	}
	delete fluidParticles;
	std::cout << "Simulation finished" << std::endl;
	std::cout << "The scene files have been saved to [build/res/assignment3]" << std::endl;

	return 0;
}
