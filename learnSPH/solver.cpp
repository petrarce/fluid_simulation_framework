#include <solver.h>
#include <kernel.h>

using namespace learnSPH::kernel;

void learnSPH::calculate_dencities(FluidSystem *fluidParticles, BorderSystem *borderParticles, Real smooth_length)
{
	auto &neighbors = fluidParticles->getNeighbors();

	assert(smooth_length > 0.0);
	assert(neighbors.size() == fluidParticles->size());
	assert(neighbors.size() == 0 || neighbors[0].size() == 2);

	auto &positions = fluidParticles->getPositions();
	auto &densities = fluidParticles->getDensities();

	auto &borderPositions = borderParticles->getPositions();
	auto &borderVolumes = borderParticles->getVolumes();

	#pragma omp parallel for schedule(guided, 100)

	for(int i = 0; i < fluidParticles->size(); i++){

		Real fluidDensity = 0.0;  // calculate density depending on neighbor fluid particles

		for(int j : neighbors[i][0]) fluidDensity += kernelFunction(positions[i], positions[j], smooth_length);

		fluidDensity += kernelFunction(positions[i], positions[i], smooth_length);  // take particle itself into account

		fluidDensity *= fluidParticles->getMass();

		Real borderDensity = 0.0;  // calculate density depending on neighbor border particles

		for(int j : neighbors[i][1]) borderDensity += kernelFunction(positions[i], borderPositions[j], smooth_length) * borderVolumes[j];

		borderDensity *= borderParticles->getRestDensity();

		densities[i] = fluidDensity + borderDensity;

		assert(densities[i] >= 0.0);
	}
}

void learnSPH::calculate_acceleration(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real viscosity, Real friction, Real stiffness, Real smooth_length)
{
	auto &neighbors = fluidParticles->getNeighbors();

	auto &densities = fluidParticles->getDensities();
	auto &positions = fluidParticles->getPositions();
	auto &velocities = fluidParticles->getVelocities();

	auto &externalForces = fluidParticles->getExternalForces();

	auto &borderPositions = borderParticles->getPositions();
	auto &borderVolumes = borderParticles->getVolumes();

	#pragma omp parallel for schedule(guided, 100)

	for(unsigned int i = 0; i < fluidParticles->size(); i++) {

		Vector3R acce_pressure_ff(0.0, 0.0, 0.0);
		Vector3R acce_viscosity_ff(0.0, 0.0, 0.0);
		Vector3R acce_pressure_fs(0.0, 0.0, 0.0);

		Real acce_viscosity_fs_factor = 0.0;

		const Real rho_i = densities[i];

		for(unsigned int j : neighbors[i][0]) {

			const Real rho_j = densities[j];

			const Vector3R grad_W_ij = kernelGradFunction(positions[i], positions[j], smooth_length);

			assert(fabs(rho_i) + threshold > threshold);
			assert(fabs(rho_j) + threshold > threshold);

			acce_pressure_ff += (max(stiffness * (rho_i - fluidParticles->getRestDensity()), 0.0) / (pow(rho_i, 2.0) + threshold) +
								max(stiffness * (rho_j - fluidParticles->getRestDensity()), 0.0) / (pow(rho_j, 2.0) + threshold)) * grad_W_ij;

			const Vector3R diff_ij = positions[i] - positions[j];
			const Vector3R diff_velo_ij = velocities[i] - velocities[j];

			acce_viscosity_ff += diff_ij.dot(grad_W_ij) / (rho_j * (diff_ij.dot(diff_ij) + 0.01 * smooth_length * smooth_length)) * diff_velo_ij;
		}
		acce_pressure_ff = fluidParticles->getMass() * acce_pressure_ff;
		acce_viscosity_ff = 2.0 * viscosity * fluidParticles->getMass() * acce_viscosity_ff;

		for(unsigned int k : neighbors[i][1]) {

			const Vector3R grad_W_ik = kernelGradFunction(positions[i], borderPositions[k], smooth_length);

			acce_pressure_fs += borderVolumes[k] * grad_W_ik;

			const Vector3R diff_ik = positions[i] - borderPositions[k];

			assert(fabs(diff_ik.dot(diff_ik) + 0.01 * smooth_length * smooth_length) > threshold);

			acce_viscosity_fs_factor += borderVolumes[k] * diff_ik.dot(grad_W_ik) / (diff_ik.dot(diff_ik) + 0.01 * smooth_length * smooth_length);
		}
		assert(pow2(rho_i) + threshold >= threshold);

		acce_pressure_fs *= fluidParticles->getRestDensity() * max(stiffness * (rho_i - fluidParticles->getRestDensity()), 0.0) / (pow2(rho_i) + threshold);

		Vector3R acce_viscosity_fs = 2.0 * friction * acce_viscosity_fs_factor * velocities[i];

		assert(fluidParticles->getMass() > threshold);

		Vector3R acce_external = 1.0 / fluidParticles->getMass() * externalForces[i];

		accelerations[i] = - acce_pressure_ff - acce_pressure_fs + acce_viscosity_ff + acce_viscosity_fs + acce_external;
    }
}

void learnSPH::symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real time_frame)
{
	auto &velocities = fluidParticles->getVelocities();
	auto &positions = fluidParticles->getPositions();

	#pragma omp parallel for schedule(guided, 100)

	for (unsigned int i = 0; i < positions.size(); i++) {
		velocities[i] += time_frame * accelerations[i];
		positions[i] += time_frame * velocities[i];
	}
}

void learnSPH::smooth_symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real epsilon, Real time_frame, Real smooth_length)
{
	auto &neighbors = fluidParticles->getNeighbors();

	auto &positions = fluidParticles->getPositions();
	auto &velocities = fluidParticles->getVelocities();
	auto &densities = fluidParticles->getDensities();

	#pragma omp parallel for schedule(guided, 100)

	for (unsigned int i = 0; i < fluidParticles->size(); i++) velocities[i] += time_frame * accelerations[i];

	vector<Vector3R> newFluidPositions(fluidParticles->size());

	#pragma omp parallel for schedule(guided, 100)

	for (unsigned int i = 0; i < fluidParticles->size(); i++) {

		Vector3R auxiliary_velocity(0.0, 0.0, 0.0);

		for(unsigned int j : neighbors[i][0]) {

			auto diff_velo_ji = velocities[j] - velocities[i];

			auxiliary_velocity += kernelFunction(positions[i], positions[j], smooth_length) / (densities[i] + densities[j] + threshold) * diff_velo_ji;
		}
		auxiliary_velocity = velocities[i] + 2.0 * epsilon * fluidParticles->getMass() * auxiliary_velocity;

		assert(auxiliary_velocity.norm() < 1e5);

		newFluidPositions[i] = positions[i] + time_frame * auxiliary_velocity;
	}
	fluidParticles->setPositions(newFluidPositions);
}
