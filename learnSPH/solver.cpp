#include <solver.h>
#include <kernel.h>

using namespace learnSPH::kernel;

void learnSPH::calculate_dencities(FluidSystem *fluidParticles, BorderSystem *borderParticles, Real smooth_length)
{
	auto &neighbors = fluidParticles->getNeighbors();

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


void learnSPH::add_press_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real stiffness, Real smooth_length)
{
	auto &neighbors = fluidParticles->getNeighbors();
	auto &densities = fluidParticles->getDensities();
	auto &positions = fluidParticles->getPositions();

	auto &borderPositions = borderParticles->getPositions();
	auto &borderVolumes = borderParticles->getVolumes();

	#pragma omp parallel for schedule(guided, 100)

	for(unsigned int i = 0; i < fluidParticles->size(); i++) {

		Vector3R acc_press_ff(0.0, 0.0, 0.0);
		Vector3R acc_press_fs(0.0, 0.0, 0.0);

		auto rho_i = densities[i];

		auto pressure_i = max(stiffness * (rho_i - fluidParticles->getRestDensity()), 0.0);

		for(unsigned int j : neighbors[i][0]) {

			auto rho_j = densities[j];

			auto pressure_j = max(stiffness * (rho_j - fluidParticles->getRestDensity()), 0.0);

			auto grad_W_ij = kernelGradFunction(positions[i], positions[j], smooth_length);

			acc_press_ff += (pressure_i / (pow2(rho_i) + threshold) + pressure_j / (pow2(rho_j) + threshold)) * grad_W_ij;
		}
		acc_press_ff *= fluidParticles->getMass();

		for(unsigned int k : neighbors[i][1]) acc_press_fs += borderVolumes[k] * kernelGradFunction(positions[i], borderPositions[k], smooth_length);

		acc_press_fs *= fluidParticles->getRestDensity() * pressure_i / (pow2(rho_i) + threshold);

		accelerations[i] -= acc_press_ff;
		accelerations[i] -= acc_press_fs;
	}
}


void learnSPH::add_visco_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real viscosity, Real friction, Real smooth_length)
{
	auto &neighbors = fluidParticles->getNeighbors();

	auto &densities = fluidParticles->getDensities();
	auto &positions = fluidParticles->getPositions();
	auto &velocities = fluidParticles->getVelocities();

	auto &borderPositions = borderParticles->getPositions();
	auto &borderVolumes = borderParticles->getVolumes();

	#pragma omp parallel for schedule(guided, 100)

	for(unsigned int i = 0; i < fluidParticles->size(); i++) {

		Vector3R acc_visco_ff(0.0, 0.0, 0.0);

		for(unsigned int j : neighbors[i][0]) {

			auto rho_j = densities[j];

			auto grad_W_ij = kernelGradFunction(positions[i], positions[j], smooth_length);

			auto diff_ij = positions[i] - positions[j];
			auto velo_ij = velocities[i] - velocities[j];

			acc_visco_ff += diff_ij.dot(grad_W_ij) / (rho_j * (diff_ij.dot(diff_ij) + 0.01 * pow2(smooth_length))) * velo_ij;
		}
		acc_visco_ff *= 2.0 * viscosity * fluidParticles->getMass();

		Real sum_visco_fs = 0.0;

		for(unsigned int k : neighbors[i][1]) {

			auto grad_W_ik = kernelGradFunction(positions[i], borderPositions[k], smooth_length);

			auto diff_ik = positions[i] - borderPositions[k];

			assert(fabs(diff_ik.dot(diff_ik) + 0.01 * pow2(smooth_length)) > threshold);

			sum_visco_fs += borderVolumes[k] * diff_ik.dot(grad_W_ik) / (diff_ik.dot(diff_ik) + 0.01 * pow2(smooth_length));
		}
		auto acc_visco_fs = 2.0 * friction * sum_visco_fs * velocities[i];

		assert(fluidParticles->getMass() > threshold);

		accelerations[i] += acc_visco_ff;
		accelerations[i] += acc_visco_fs;
    }
}

void learnSPH::add_exter_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles)
{
	auto &externalForces = fluidParticles->getExternalForces();

	#pragma omp parallel for schedule(guided, 100)

	for(unsigned int i = 0; i < fluidParticles->size(); i++) accelerations[i] += 1.0 / fluidParticles->getMass() * externalForces[i];
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
