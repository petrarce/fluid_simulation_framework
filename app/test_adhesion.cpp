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
	assert(argc == 23);

	std::cout << "Simulation running" << std::endl;

	Vector3R lowerCorner(stod(argv[1]), stod(argv[2]), stod(argv[3]));
	Vector3R upperCorner(stod(argv[4]), stod(argv[5]), stod(argv[6]));

	Vector3R center(stod(argv[7]), stod(argv[8]), stod(argv[9]));
	
	Real radius = stod(argv[10]);

	Real sampling_distance = stod(argv[11]);
	Real eta = stod(argv[12]);

	size_t n_iterations = stod(argv[13]);

	Real multiplier = stod(argv[14]);

	Real viscosity = stod(argv[15]);
	Real friction = stod(argv[16]);

	Real gamma = stod(argv[17]);
	Real beta = stod(argv[18]);

	Real gravity = stod(argv[19]);

	Real render_step = stod(argv[20]);
	Real sim_duration = stod(argv[21]);
	string sim_name = argv[22];

	FluidSystem* fluidParticles = sample_fluid_cube(lowerCorner, upperCorner, 1000.0, sampling_distance, eta);

	cout << "Number of fluid particles: " << fluidParticles->size() << endl;

	BorderSystem* borderParticles = sample_border_sphere(radius, center, 3000.0, sampling_distance * 0.5, eta);

	cout << "Number of border particles: " << borderParticles->size() << endl;

	NeighborhoodSearch ns(fluidParticles->getCompactSupport());

	ns.add_point_set((Real*)fluidParticles->getPositions().data(), fluidParticles->size(), true);

	ns.add_point_set((Real*)borderParticles->getPositions().data(), borderParticles->size(), false);

	fluidParticles->setGravity(gravity);

	int n_frames = sim_duration / render_step;

	cout << "the simulation lasts [" << sim_duration << "] seconds consisting of [" << n_frames << "] frames. a frame is rendered every [" << render_step << "] seconds" << endl;

	string filename = "res/assignment4/border.vtk";

	vector<Vector3R> dummyVector(borderParticles->size());

	learnSPH::saveParticlesToVTK(filename, borderParticles->getPositions(), borderParticles->getVolumes(), dummyVector);

	fluidParticles->findNeighbors(ns);

	for (int frame = 0; frame < n_frames; frame ++) {

		Real cur_sim_time = 0.0;

		int physical_steps = 0;

		while (cur_sim_time < render_step) {

			learnSPH::calculate_dencities(fluidParticles, borderParticles);

			vector<Vector3R> accelerations(fluidParticles->size(), Vector3R(0.0, 0.0, 0.0));

			learnSPH::add_visco_component(accelerations, fluidParticles, borderParticles, viscosity, friction);

			learnSPH::add_exter_component(accelerations, fluidParticles);

			learnSPH::add_surfa_component(accelerations, fluidParticles, borderParticles, gamma, beta);

			Real update_step = min(render_step - cur_sim_time, fluidParticles->getCourantBound());

			auto positions = fluidParticles->getPositions();

			learnSPH::smooth_symplectic_euler(accelerations, fluidParticles, 0.5, update_step);

			fluidParticles->findNeighbors(ns);

			learnSPH::correct_position(fluidParticles, borderParticles, positions, update_step, multiplier, n_iterations);

			fluidParticles->clipVelocities(50.0);

			cur_sim_time += update_step;

			physical_steps ++;
		}
		cout << "\n[" << physical_steps << "] physical updates were carried out for rendering frame [" << frame << "] / [" << n_frames << "]" << endl;

		string filename = "res/assignment4/" + sim_name + '_' + std::to_string(frame) + ".vtk";

		learnSPH::saveParticlesToVTK(filename, fluidParticles->getPositions(), fluidParticles->getDensities(), fluidParticles->getVelocities());

		vector<Real> params;

		params.push_back(fluidParticles->getCompactSupport());
		params.push_back(fluidParticles->getSmoothLength());
		params.push_back(fluidParticles->getMass());
		params.push_back(fluidParticles->getRestDensity());

		filename = "res/assignment4/" + sim_name + "_params_" + std::to_string(frame) + ".cereal";

		save_scalars(filename, params);

		filename = "res/assignment4/" + sim_name + "_positions_" + std::to_string(frame) + ".cereal";

		save_vectors(filename, fluidParticles->getPositions());

		filename = "res/assignment4/" + sim_name + "_densities_" + std::to_string(frame) + ".cereal";

		save_scalars(filename, fluidParticles->getDensities());
	}
	delete fluidParticles;
	std::cout << "Simulation finished" << std::endl;
	std::cout << "The scene files have been saved to [build/res/assignment4]" << std::endl;

	return 0;
}
