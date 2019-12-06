#pragma once
#include <storage.h>
#include <vector>

using namespace std;
using namespace learnSPH;

namespace learnSPH{

	void calculate_dencities(FluidSystem *fluidParticles, BorderSystem *borderParticles, Real smooth_length);

	void add_press_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real stiffness, Real smooth_length);

	void add_visco_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real viscosity, Real friction, Real smooth_length);

	void add_exter_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles);

	void symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real time_frame);

	void smooth_symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real epsilon, Real time_frame, Real smooth_length);
};
