#include <solver.h>
#include <kernel.h>
opcode Solver::calculate_dencities(NormalPartDataSet& fluidParticles,
	const BorderPartDataSet& borderParticles,
	const vector<vector<vector<unsigned int>>>& fluidParticleNeighbours,
	const Real smoothingLength)
{
	assert(smoothingLength > 0.0);
	//fluidParticleNeighbours should contain set of neighbors for each particle of the fluid
	assert(fluidParticleNeighbours.size() == fluidParticles.getNumberOfParticles());
	//for each particle there should be two sets of neighbors: 
	//	fluid neighbors set and border neighbor set
	assert(fluidParticleNeighbours.size() == 0 || fluidParticleNeighbours[0].size() == 2);

	vector<Real>& fluidParticlesDensities = fluidParticles.getParticleDencities();
	const Vector3R* fluidParticlesPositions = fluidParticles.getParticlePositionsData();
	const Vector3R* borderParticlePositions = borderParticles.getParticlePositionsData();
	const Real* borderParticlesVolumes = borderParticles.getParticleVolumeData();


	#pragma omp parallel for schedule(guided, 100)
	for(int i = 0; i < fluidParticles.getNumberOfParticles(); i++){
		//calculate density depending on neighbor fluid particles
		Real fluidDensity = 0.0;
		for(int j : fluidParticleNeighbours[i][0]){
			fluidDensity += learnSPH::kernel::kernelFunction(fluidParticlesPositions[i], 
										fluidParticlesPositions[j], 
										smoothingLength);
		}
		fluidDensity *= fluidParticles.getParticleMass();

		Real borderDensity = 0.0;
		//calculate density depending on neighbor border particles
		for(int j : fluidParticleNeighbours[i][1]){
			borderDensity += learnSPH::kernel::kernelFunction(fluidParticlesPositions[i], 
										borderParticlePositions[j], 
										smoothingLength)*borderParticlesVolumes[j];
		}
		borderDensity *= borderParticles.getRestDensity();

		fluidParticlesDensities[i] = fluidDensity + borderDensity;
	}
	return STATUS_OK;
}

// set NormalPartDataSet as constant, switch back if causing mistakes.
opcode Solver::calculate_acceleration(vector<Vector3R>& fluidParticlesAccelerations,
                                      const NormalPartDataSet& fluidParticles,
                                      const BorderPartDataSet& borderParticles,
                                      const vector<vector<vector<unsigned int>>>& fluidParticleNeighbours,
                                      const Real fluid_viscosity,
                                      const Real friction_para,
                                      const Real stiffness_para,
                                      const Real smoothingLength) {
    const Real* fluidParticlesDensities = fluidParticles.getParticleDencitiesData();
    const Vector3R* fluidParticlesPositions = fluidParticles.getParticlePositionsData();
    const Vector3R* borderParticlePositions = borderParticles.getParticlePositionsData();
    const Vector3R* fluidParticlesVelocities = fluidParticles.getParticleVelocitiesData();
    const Vector3R* fluidParticlesForces = fluidParticles.getParticleForcesdata();
    const Real* borderParticlesVolumes = borderParticles.getParticleVolumeData();


    #pragma omp parallel for schedule(guided, 100)
    for(unsigned int i = 0; i < fluidParticles.getNumberOfParticles(); i++){
        // fluid-fluid interactions
        Vector3R acce_pressure_ff(0.0,0.0,0.0);
        Vector3R acce_viscosity_ff(0.0,0.0,0.0);
        // fluid-solid interaction
        Vector3R acce_pressure_fs(0.0,0.0,0.0);
        //Vector3R acce_viscosity_fs(0.0,0.0,0.0);
        Real acce_viscosity_fs_factor = 0.0;

        const Real rho_i = fluidParticlesDensities[i];
        for(unsigned int j : fluidParticleNeighbours[i][0]){
            const Real rho_j = fluidParticlesDensities[j];
            //
            const Vector3R grad_W_ij = learnSPH::kernel::kernelGradFunction(fluidParticlesPositions[i],
                                                                             fluidParticlesPositions[j],
                                                                             smoothingLength);
            acce_pressure_ff +=(max(stiffness_para*(rho_i-fluidParticles.getRestDensity()),0.0)/pow(rho_i,2.0)+
                                max(stiffness_para*(rho_j-fluidParticles.getRestDensity()),0.0)/pow(rho_j,2.0))*
                                grad_W_ij;

            const Vector3R diff_ij = fluidParticlesPositions[i]-fluidParticlesPositions[j];
            acce_viscosity_ff += diff_ij.dot(grad_W_ij)/
                    (rho_j*(diff_ij.dot(diff_ij)+0.01*smoothingLength*smoothingLength))*
                    (fluidParticlesVelocities[i]-fluidParticlesVelocities[j]);
        }
        acce_pressure_ff = fluidParticles.getParticleMass()*acce_pressure_ff;
        acce_viscosity_ff = 2.0*fluid_viscosity*fluidParticles.getParticleMass()*acce_viscosity_ff;
        for(unsigned int k: fluidParticleNeighbours[i][1]){
            const Vector3R grad_W_ik = learnSPH::kernel::kernelGradFunction(fluidParticlesPositions[i],
                                                                            borderParticlePositions[k],
                                                                            smoothingLength);
            //
            acce_pressure_fs += borderParticlesVolumes[k]*grad_W_ik;

            const Vector3R diff_ik =fluidParticlesPositions[i]-borderParticlePositions[k];
            acce_viscosity_fs_factor += borderParticlesVolumes[k]*
                    diff_ik.dot(grad_W_ik)/(diff_ik.dot(diff_ik) + 0.01*smoothingLength*smoothingLength);

        }
        acce_pressure_fs *= fluidParticles.getRestDensity()*
                            max(stiffness_para*(rho_i-fluidParticles.getRestDensity()),0.0)/pow(rho_i,2.0);

        Vector3R acce_viscosity_fs = 2.0*friction_para*acce_viscosity_fs_factor * fluidParticlesVelocities[i];

        Vector3R acce_external =1.0 / fluidParticles.getParticleMass() * fluidParticlesForces[i];

        Vector3R acceleration = - acce_pressure_ff - acce_pressure_fs
                                + acce_viscosity_ff + acce_viscosity_fs
                                + acce_external;
        fluidParticlesAccelerations[i] = acceleration;
    }
    return STATUS_OK;
}

opcode Solver::semi_implicit_Euler(const vector<Vector3R> &fluidParticlesAccelerations,
                                   NormalPartDataSet& fluidParticles, const Real time_frame) {
    vector<Vector3R>& fluidParticlesVelocities  = fluidParticles.getParticleVelocities();
    vector<Vector3R>& fluidParticlesPositions = fluidParticles.getParticlePositions();

    #pragma omp parallel for schedule(guided, 100)
    for (unsigned int i=0; i< fluidParticlesPositions.size(); i++){
        fluidParticlesVelocities[i] += time_frame * fluidParticlesAccelerations[i];
        fluidParticlesPositions[i] += time_frame * fluidParticlesVelocities[i];
    }
    return STATUS_OK;
}

opcode Solver::mod_semi_implicit_Euler(const vector<Vector3R> &fluidParticlesAccelerations,
                                       NormalPartDataSet &fluidParticles,
                                       const vector<vector<vector<unsigned int>>> &normalParticleNeighbours,
                                       const Real scaling_para, const Real time_frame,
                                       const Real smoothingLengthFactor) {
    vector<Vector3R>& fluidParticlesVelocities  = fluidParticles.getParticleVelocities();
    vector<Vector3R>& fluidParticlesPositions = fluidParticles.getParticlePositions();
    const Real* fluidParticlesDensities = fluidParticles.getParticleDencitiesData();
    #pragma omp parallel for schedule(guided, 100)
    for (unsigned int i=0; i< fluidParticlesPositions.size(); i++){
        fluidParticlesVelocities[i] += time_frame * fluidParticlesAccelerations[i];
    }

    #pragma omp parallel for schedule(guided, 100)
    for (unsigned int i=0; i< fluidParticlesPositions.size(); i++){
        Vector3R auxiliary_velocity(0.0,0.0,0);
        for(unsigned int j : normalParticleNeighbours[i][0]){
            auxiliary_velocity += (learnSPH::kernel::kernelFunction(fluidParticlesPositions[i], fluidParticlesPositions[j], smoothingLengthFactor)/
                    (fluidParticlesDensities[i] + fluidParticlesDensities[j]))*(fluidParticlesVelocities[j]-fluidParticlesVelocities[i]);
        }
        auxiliary_velocity = fluidParticlesVelocities[i] + 2.0 * scaling_para * fluidParticles.getParticleMass() * auxiliary_velocity;
        fluidParticlesPositions[i] += time_frame * auxiliary_velocity;
    }
    return STATUS_OK;
}
	
