#include <solver.h>
#include <kernel.h>

using namespace learnSPH::kernel;

void learnSPH::calculate_dencities(
								FluidSystem *fluidParticles,
								BorderSystem *borderParticles,
								const vector<vector<vector<unsigned int> > >& fluidParticleNeighbours,
								const Real smoothingLength)
{
	assert(smoothingLength > 0.0);
	assert(fluidParticleNeighbours.size() == fluidParticles->size());
	assert(fluidParticleNeighbours.size() == 0 || fluidParticleNeighbours[0].size() == 2);

	vector<Real>& fluidParticlesDensities = fluidParticles->getDensities();

	auto fluidParticlesPositions = fluidParticles->getPositions();
	auto borderParticlePositions = borderParticles->getPositions();
	auto borderParticlesVolumes = borderParticles->getVolumes();

	#pragma omp parallel for schedule(guided, 100)

	for(int i = 0; i < fluidParticles->size(); i++){

		Real fluidDensity = 0.0;  // calculate density depending on neighbor fluid particles

		for(int j : fluidParticleNeighbours[i][0]) fluidDensity += kernelFunction(fluidParticlesPositions[i], fluidParticlesPositions[j], smoothingLength);

		fluidDensity *= fluidParticles->getMass();

		Real borderDensity = 0.0;  // calculate density depending on neighbor border particles

		for(int j : fluidParticleNeighbours[i][1]) borderDensity += kernelFunction(fluidParticlesPositions[i], borderParticlePositions[j], smoothingLength) * borderParticlesVolumes[j];

		borderDensity *= borderParticles->getRestDensity();

		fluidParticlesDensities[i] = fluidDensity + borderDensity;

		assert(fluidParticlesDensities[i] >= 0.0);
	}
}

void learnSPH::calculate_acceleration(
									vector<Vector3R>& fluidParticlesAccelerations,
									FluidSystem *fluidParticles,
									BorderSystem *borderParticles,
									const vector<vector<vector<unsigned int> > >& fluidParticleNeighbours,
									const Real fluid_viscosity,
									const Real friction_para,
									const Real stiffness_para,
									const Real smoothingLength)
{
	auto fluidParticlesDensities = fluidParticles->getDensities();
	auto fluidParticlesPositions = fluidParticles->getPositions();
	auto borderParticlePositions = borderParticles->getPositions();
	auto fluidParticlesVelocities = fluidParticles->getVelocities();
	auto fluidParticlesForces = fluidParticles->getExternalForces();
	auto borderParticlesVolumes = borderParticles->getVolumes();

	#pragma omp parallel for schedule(guided, 100)

	for(unsigned int i = 0; i < fluidParticles->size(); i++) {

		Vector3R acce_pressure_ff(0.0, 0.0, 0.0);  // fluid-fluid interactions
		Vector3R acce_viscosity_ff(0.0, 0.0, 0.0);  // fluid-fluid frictions
		Vector3R acce_pressure_fs(0.0, 0.0, 0.0);  // fluid-solid interactions

		Real acce_viscosity_fs_factor = 0.0;  // fluid-solid frictions

		const Real rho_i = fluidParticlesDensities[i];

		for(unsigned int j : fluidParticleNeighbours[i][0]) {

			const Real rho_j = fluidParticlesDensities[j];

			const Vector3R grad_W_ij = kernelGradFunction(fluidParticlesPositions[i], fluidParticlesPositions[j], smoothingLength);

			assert(fabs(rho_i) + threshold > threshold);
			assert(fabs(rho_j) + threshold > threshold);

			acce_pressure_ff += (max(stiffness_para * (rho_i - fluidParticles->getRestDensity()), 0.0) / (pow(rho_i, 2.0) + threshold) +
								max(stiffness_para * (rho_j - fluidParticles->getRestDensity()), 0.0) / (pow(rho_j, 2.0) + threshold)) * grad_W_ij;

			const Vector3R diff_ij = fluidParticlesPositions[i] - fluidParticlesPositions[j];
			const Vector3R diff_velo_ij = fluidParticlesVelocities[i] - fluidParticlesVelocities[j];

			acce_viscosity_ff += diff_ij.dot(grad_W_ij) / (rho_j * (diff_ij.dot(diff_ij) + 0.01 * smoothingLength * smoothingLength)) * diff_velo_ij;
		}
		acce_pressure_ff = fluidParticles->getMass() * acce_pressure_ff;
		acce_viscosity_ff = 2.0 * fluid_viscosity * fluidParticles->getMass() * acce_viscosity_ff;

		for(unsigned int k : fluidParticleNeighbours[i][1]) {

			const Vector3R grad_W_ik = kernelGradFunction(fluidParticlesPositions[i], borderParticlePositions[k], smoothingLength);

			acce_pressure_fs += borderParticlesVolumes[k] * grad_W_ik;

			const Vector3R diff_ik = fluidParticlesPositions[i] - borderParticlePositions[k];

			assert(fabs(diff_ik.dot(diff_ik) + 0.01 * smoothingLength * smoothingLength) > threshold);

			acce_viscosity_fs_factor += borderParticlesVolumes[k] * diff_ik.dot(grad_W_ik) / (diff_ik.dot(diff_ik) + 0.01 * smoothingLength * smoothingLength);
		}
		assert(pow2(rho_i) + threshold >= threshold);

		acce_pressure_fs *= fluidParticles->getRestDensity() * max(stiffness_para * (rho_i - fluidParticles->getRestDensity()), 0.0) / (pow2(rho_i) + threshold);

		Vector3R acce_viscosity_fs = 2.0 * friction_para * acce_viscosity_fs_factor * fluidParticlesVelocities[i];

		assert(fluidParticles->getMass() > threshold);

		Vector3R acce_external = 1.0 / fluidParticles->getMass() * fluidParticlesForces[i];

		fluidParticlesAccelerations[i] = - acce_pressure_ff - acce_pressure_fs + acce_viscosity_ff + acce_viscosity_fs + acce_external;
    }
}

void learnSPH::symplectic_euler(
							const vector<Vector3R> &fluidParticlesAccelerations,
							FluidSystem *fluidParticles,
							const Real time_frame)
{
	vector<Vector3R>& fluidParticlesVelocities = fluidParticles->getVelocities();
	vector<Vector3R>& fluidParticlesPositions = fluidParticles->getPositions();

	#pragma omp parallel for schedule(guided, 100)

	for (unsigned int i = 0; i < fluidParticlesPositions.size(); i++) {
		fluidParticlesVelocities[i] += time_frame * fluidParticlesAccelerations[i];
		fluidParticlesPositions[i] += time_frame * fluidParticlesVelocities[i];
	}
}

void learnSPH::smooth_symplectic_euler(
									const vector<Vector3R> &fluidParticlesAccelerations,
									FluidSystem *fluidParticles,
									const vector<vector<vector<unsigned int> > >& normalParticleNeighbours,
									const Real scaling_para,
									const Real time_frame,
									const Real smoothingLengthFactor)
{
	vector<Vector3R>& fluidParticlesVelocities = fluidParticles->getVelocities();
	vector<Vector3R>& fluidParticlesPositions = fluidParticles->getPositions();

	auto fluidParticlesDensities = fluidParticles->getDensities();
	
	#pragma omp parallel for schedule(guided, 100)

	for (unsigned int i = 0; i < fluidParticlesPositions.size(); i++) fluidParticlesVelocities[i] += time_frame * fluidParticlesAccelerations[i];

	vector<Vector3R> newFluidPositions(fluidParticlesPositions.size());

	#pragma omp parallel for schedule(guided, 100)

	for (unsigned int i = 0; i < fluidParticlesPositions.size(); i++) {

		Vector3R auxiliary_velocity(0.0, 0.0, 0.0);

		for(unsigned int j : normalParticleNeighbours[i][0]) {

			auto diff_velo_ji = fluidParticlesVelocities[j] - fluidParticlesVelocities[i];

			auxiliary_velocity += kernelFunction(fluidParticlesPositions[i], fluidParticlesPositions[j], smoothingLengthFactor) / (fluidParticlesDensities[i] + fluidParticlesDensities[j] + threshold) * diff_velo_ji;
		}
		auxiliary_velocity = fluidParticlesVelocities[i] + 2.0 * scaling_para * fluidParticles->getMass() * auxiliary_velocity;

		assert(auxiliary_velocity.norm() < 1e5);

		newFluidPositions[i] = fluidParticlesPositions[i] + time_frame * auxiliary_velocity;
	}
	fluidParticles->setPositions(newFluidPositions);
}
