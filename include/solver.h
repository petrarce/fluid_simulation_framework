#pragma once
#include <data_set.h>
#include <vector>

using namespace std;
using namespace learnSPH;

namespace learnSPH{

	void calculate_dencities(
						NormalPartDataSet& normalParticles,
						BorderPartDataSet& borderParticles,
						const vector<vector<vector<unsigned int> > >& normalParticleNeighbours,
						const Real smoothingLengthFactor = 1);

	void calculate_acceleration(
						vector<Vector3R>& fluidParticlesAccelerations,
						NormalPartDataSet& fluidParticles,
						BorderPartDataSet& borderParticles,
						const vector<vector<vector<unsigned int> > >& normalParticleNeighbours,
						const Real fluid_viscosity,
						const Real friction_para,
						const Real stiffness_para,
						const Real smoothingLength);

	void symplectic_euler(
						const vector<Vector3R>& fluidParticlesAccelerations,
						NormalPartDataSet& normalParticles,
						const Real time_frame);

	void smooth_symplectic_euler(
						const vector<Vector3R>& fluidParticlesAccelerations,
						NormalPartDataSet& fluidParticles,
						const vector<vector<vector<unsigned int> > >& normalParticleNeighbours,
						const Real scaling_para,
						const Real time_frame,
						const Real smoothingLength);
};