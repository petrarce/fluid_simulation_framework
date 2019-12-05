#pragma once
#include <storage.h>
#include <vector>

using namespace std;
using namespace learnSPH;

namespace learnSPH{

	void calculate_dencities(FluidSystem *fluidParticles, BorderSystem *borderParticles, Real smooth_length);

	void calculate_acceleration(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real viscosity, Real friction, Real stiffness, Real smooth_length);

	void symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real time_frame);

	void smooth_symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real epsilon, Real time_frame, Real smooth_length);
};
