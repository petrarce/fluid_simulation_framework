#pragma once
#include <storage.h>
#include <vector>

using namespace std;
using namespace learnSPH;

namespace learnSPH{

	void calculate_dencities(
						FluidSystem *normalParticles,
						BorderSystem *borderParticles,
						const vector<vector<vector<unsigned int> > >& normalParticleNeighbours,
						const Real smoothingLengthFactor = 1);

	void calculate_acceleration(
						vector<Vector3R>& fluidParticlesAccelerations,
						FluidSystem *fluidParticles,
						BorderSystem *borderParticles,
						const vector<vector<vector<unsigned int> > >& normalParticleNeighbours,
						const Real fluid_viscosity,
						const Real friction_para,
						const Real stiffness_para,
						const Real smoothingLength);

	void symplectic_euler(
						const vector<Vector3R>& fluidParticlesAccelerations,
						FluidSystem *normalParticles,
						const Real time_frame);

	void smooth_symplectic_euler(
						const vector<Vector3R>& fluidParticlesAccelerations,
						FluidSystem *fluidParticles,
						const vector<vector<vector<unsigned int> > >& normalParticleNeighbours,
						const Real scaling_para,
						const Real time_frame,
						const Real smoothingLength);
};