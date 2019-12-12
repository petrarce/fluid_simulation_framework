#include <solver.h>
#include <kernel.h>

using namespace learnSPH::kernel;

void learnSPH::calculate_dencities(FluidSystem *fluidParticles, BorderSystem *borderParticles)
{
	auto smooth_length = fluidParticles->getSmoothingLength();

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


void learnSPH::add_press_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real stiffness)
{
	auto smooth_length = fluidParticles->getSmoothingLength();

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


void learnSPH::add_visco_component(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, BorderSystem *borderParticles, Real viscosity, Real friction)
{
	auto smooth_length = fluidParticles->getSmoothingLength();

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


void learnSPH::symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real delta_t)
{
	auto &velocities = fluidParticles->getVelocities();
	auto &positions = fluidParticles->getPositions();

	#pragma omp parallel for schedule(guided, 100)

	for (unsigned int i = 0; i < positions.size(); i++) {
		velocities[i] += delta_t * accelerations[i];
		positions[i] += delta_t * velocities[i];
	}
}


void learnSPH::smooth_symplectic_euler(vector<Vector3R> &accelerations, FluidSystem *fluidParticles, Real epsilon, Real delta_t)
{
	auto smooth_length = fluidParticles->getSmoothingLength();

	auto &neighbors = fluidParticles->getNeighbors();

	auto &positions = fluidParticles->getPositions();
	auto &velocities = fluidParticles->getVelocities();
	auto &densities = fluidParticles->getDensities();

	#pragma omp parallel for schedule(guided, 100)

	for (unsigned int i = 0; i < fluidParticles->size(); i++) velocities[i] += delta_t * accelerations[i];

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

		newFluidPositions[i] = positions[i] + delta_t * auxiliary_velocity;
	}
	fluidParticles->setPositions(newFluidPositions);
}



void learnSPH::correct_position(FluidSystem *fluidParticles, BorderSystem *borderParticles, Real delta_t,
                                    size_t n_iterations, NeighborhoodSearch &ns) {
    Real epsilon = 1e-4;
    // make copy of neighbors of each particles <vector<int>> fixedNeighbors
//    vector<vector<vector<unsigned int> > > fixedNeighbors(fluidParticles->getNeighbors());
    vector<vector<vector<unsigned int> > >& neighbors = fluidParticles->getNeighbors();
    // vector<Vector3R> neighborFixedParticlePositions
//    vector<Vector3R> neighborFixedParticlePositions(fluidParticles->getPositions());
    auto &positions = fluidParticles->getPositions();
    auto &borderPositions = borderParticles->getPositions();
    auto &velocities = fluidParticles->getVelocities();
    // Vector3R originalParticlePositions
    vector<Vector3R> originalParticlePositions(fluidParticles->getPositions());

    auto &volumes = borderParticles->getVolumes();

    size_t current_iterations = 0;
    while(current_iterations < n_iterations){ // terminatal condition
        fluidParticles->findNeighbors(ns);
        // All following steps include the neighbor-fixed and neighbor-flexible versions
        //density estimation
        learnSPH::calculate_dencities(fluidParticles, borderParticles);
        auto &densities = fluidParticles->getDensities();
        //compute S_i
        vector<Real> S_i(fluidParticles->size());
        #pragma omp parallel for schedule(guided, 100)
        for(unsigned int i=0;i < fluidParticles->size(); i++){
            Vector3R term1{0,0,0};
            Real term2 = 0.0;
            for(unsigned int j : neighbors[i][0]){
                term1 += learnSPH::kernel::kernelGradFunction(positions[i], positions[j], fluidParticles->getSmoothingLength());
                Vector3R term2i = learnSPH::kernel::kernelGradFunction(positions[i], positions[j], fluidParticles->getSmoothingLength());
                term2 += sqrt(term2i.dot(term2i));
            }
            term1 = fluidParticles->getMass()/fluidParticles->getRestDensity()*term1;
            term2 = term2/fluidParticles->getRestDensity();
            for(unsigned int k : neighbors[i][1]){
                term1 += volumes[k]*learnSPH::kernel::kernelGradFunction(positions[i], borderPositions[k], fluidParticles->getSmoothingLength());
            }
            S_i[i] = sqrt(term1.dot(term1))/fluidParticles->getMass() + term2;
        }
        //compute lambda_i
        vector<Real> lambda_i(fluidParticles->size());
        #pragma omp parallel for schedule(guided, 100)
        for(unsigned int i=0; i<fluidParticles->size();i++){
            Real C_i = densities[i]/fluidParticles->getRestDensity() -1;
            lambda_i[i] = C_i > 0 ? (-C_i/(S_i[i]+epsilon)) : 0;
        }
        //compute Delta_xi
        vector<Vector3R> Delta_xi(fluidParticles->size());
        #pragma omp parallel for schedule(guided, 100)
        for(unsigned int i=0; i< fluidParticles->size(); i++){
            Vector3R term1{0,0,0};
            Vector3R term2{0,0,0};
            for(unsigned int j: neighbors[i][0]){
                term1 += (lambda_i[i] + lambda_i[j]) * learnSPH::kernel::kernelGradFunction(positions[i], positions[j], fluidParticles->getSmoothingLength());
            }
            term1 = term1/fluidParticles->getRestDensity();
            for(unsigned int k: neighbors[i][1]){
                term2 += volumes[k] * learnSPH::kernel::kernelGradFunction(positions[i], borderPositions[k], fluidParticles->getSmoothingLength());
            }
            term2 = lambda_i[i]/fluidParticles->getMass() * term2;
            Delta_xi[i] = term1 + term2;
        }
        //update Xi
        #pragma omp parallel for schedule(guided, 100)
        for(unsigned int i=0; i< fluidParticles->size(); i++){
            positions[i] += Delta_xi[i];
        }
        current_iterations ++;
    }
    #pragma omp parallel for schedule(guided, 100)
    for(unsigned int i=0; i< fluidParticles->size(); i++){
        velocities[i] = (positions[i] - originalParticlePositions[i])/delta_t;
    }
    return;
}
