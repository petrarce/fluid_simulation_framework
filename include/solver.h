#pragma once
#include <storage.h>
#include <vector>

using namespace std;
using namespace learnSPH;

namespace learnSPH{

	void calculate_dencities(FluidSystem *fluidParticles, BorderSystem *borderParticles);

	void add_press_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real stiffness);

	void add_visco_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real viscosity, Real friction);

	void add_exter_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles);

	void symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real delta_t);

	void smooth_symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real epsilon, Real delta_t);

	void correct_position(FluidSystem *fluidParticles, BorderSystem *borderParticles, Real delta_t, size_t n_iterations, vector<Vector3R> &prev_pos);

	/*
	 * The develop version of correct_position
	 * used to decide whether the particle positions should be updated during the Newton-like solver.
	 * trace the following statistics:
	 *  - Relative difference between update and non-update procedure wrt the non-update delta_Xi
	 *  - Ratio of changes in neighbors (both deletions and insertions)
	 */
	void correct_position_dev(FluidSystem *fluidParticles, BorderSystem *borderParticles, Real delta_t, size_t n_iterations, NeighborhoodSearch &ns);
};
