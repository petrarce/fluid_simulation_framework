#pragma once
#include <data_set.h>
#include <vector>

using namespace std;
using namespace learnSPH;

namespace learnSPH{
	class Solver {
	private:
	public:
		static opcode calculate_dencities(NormalPartDataSet& normalParticles,
			const BorderPartDataSet& borderParticles,
			const vector<vector<vector<unsigned int>>>& normalParticleNeighbours,
			const Real smoothingLengthFactor = 1);
	    static opcode calculate_acceleration(vector<Vector3R>& fluidParticlesAccelerations,
	                                         const NormalPartDataSet& fluidParticles,
                                             const BorderPartDataSet& borderParticles,
                                             const vector<vector<vector<unsigned int>>>& normalParticleNeighbours,
                                             const Real fluid_viscosity,
                                             const Real friction_para,
                                             const Real stiffness_para = 1000,
                                             const Real smoothingLengthFactor = 1);

	    static opcode semi_implicit_Euler(const vector<Vector3R>& fluidParticlesAccelerations,
                                          NormalPartDataSet& normalParticles,
                                          const Real time_frame);

	    static opcode mod_semi_implicit_Euler(const vector<Vector3R>& fluidParticlesAccelerations,
                                                     NormalPartDataSet& fluidParticles,
                                                     const vector<vector<vector<unsigned int>>>& normalParticleNeighbours,
                                                     const Real scaling_para,
                                                     const Real time_frame,
                                                     const Real smoothingLengthFactor = 1);
	
	private:
		//deleted constructors and operator=
		Solver();
		Solver(const Solver&);
		~Solver();
		const Solver& operator=(const Solver&){return *this;};
		unsigned int curr_timestep;
	};
};