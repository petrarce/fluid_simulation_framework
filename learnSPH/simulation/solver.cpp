#include <learnSPH/simulation/solver.h>
#include <learnSPH/core/kernel.h>

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


void learnSPH::correct_position(FluidSystem *fluidParticles, 
								BorderSystem *borderParticles, 
								vector<Vector3R> &prev_pos, 
								Real delta_t, 
								size_t n_iterations,
								Real velocityMultiplier)
{
    auto smooth_length = fluidParticles->getSmoothingLength();

    auto &neighbors = fluidParticles->getNeighbors();
    auto &positions = fluidParticles->getPositions();
    auto &velocities = fluidParticles->getVelocities();

    auto &borderPositions = borderParticles->getPositions();
    auto &borderVolumes = borderParticles->getVolumes();

    size_t current_iterations = 0;

    vector<Real> denominator(fluidParticles->size());
    vector<Real> lambda(fluidParticles->size());
    vector<Vector3R> deltaX(fluidParticles->size());

    while(current_iterations < n_iterations) {

        learnSPH::calculate_dencities(fluidParticles, borderParticles);

        auto &densities = fluidParticles->getDensities();

        #pragma omp parallel for schedule(guided, 100)

        for(unsigned int i = 0; i < fluidParticles->size(); i++) {
           assert(fluidParticles->getRestDensity() > threshold);
            Real C_i = max(densities[i] / fluidParticles->getRestDensity() - 1.0, 0.0);
        	if(C_i > threshold){

	            auto term1 = Vector3R(0.0, 0.0, 0.0);
	            auto term2 = Vector3R(0.0, 0.0, 0.0);

	            Real term3 = 0.0;

	            for(unsigned int j : neighbors[i][0]) {

	                auto grad_W_ij = kernelGradFunction(positions[i], positions[j], smooth_length);

	                auto term = grad_W_ij;

	                term1 += term;

	                term3 += term.dot(term);
	            }
	            assert(fluidParticles->getRestDensity() > threshold);
	            term1 *= fluidParticles->getMass() / fluidParticles->getRestDensity();
	            term3 *= pow2(fluidParticles->getMass() / fluidParticles->getRestDensity());

	            for(unsigned int k : neighbors[i][1]) {

	                auto grad_W_ik = kernelGradFunction(positions[i], borderPositions[k], smooth_length);

	                term2 += borderVolumes[k] * grad_W_ik;
	            }
	            assert(fluidParticles->getMass() > threshold);
	            denominator[i] = ((term1 + term2).dot(term1 + term2) + term3) / fluidParticles->getMass();

	            lambda[i] = -C_i  / (denominator[i] + 1e-4);
	        } else {
	        	lambda[i] = 0;
	        }
        }

        #pragma omp parallel for schedule(guided, 100)

        for(unsigned int i = 0; i < fluidParticles->size(); i++){

            auto term1 = Vector3R(0.0, 0.0, 0.0);
            auto term2 = Vector3R(0.0, 0.0, 0.0);

            for(unsigned int j : neighbors[i][0]) term1 += (lambda[i] + lambda[j]) * kernelGradFunction(positions[i], positions[j], smooth_length);
            assert(fluidParticles->getRestDensity() > threshold);
            term1 /= fluidParticles->getRestDensity();

            for(unsigned int k : neighbors[i][1]) term2 += borderVolumes[k] * kernelGradFunction(positions[i], borderPositions[k], smooth_length);

            assert(fluidParticles->getMass() > threshold);
            term2 = lambda[i] / fluidParticles->getMass() * term2;

            deltaX[i] = term1 + term2;
        }

        #pragma omp parallel for schedule(guided, 100)

        for(unsigned int i = 0; i < fluidParticles->size(); i++) positions[i] += deltaX[i];

        current_iterations ++;
    }

    #pragma omp parallel for schedule(guided, 100)
    for(unsigned int i = 0; i < fluidParticles->size(); i++){
    	velocities[i] = (1-velocityMultiplier)*velocities[i] + 
    						velocityMultiplier*(positions[i] - prev_pos[i]) / delta_t;
    }
}

void learnSPH::add_surface_tension_component(vector<Vector3R>& accelerations,
								const FluidSystem *fluidParticles,
								const BorderSystem *borderParticles,
								Real lambda,
								Real betha)
{
	const vector<vector<vector<unsigned int>>>& neighbors = fluidParticles->getNeighbors();
	const vector<Real>& densities = fluidParticles->getDensities();
	const vector<Vector3R>& positions = fluidParticles->getPositions();
	const vector<Vector3R>& borderPositions = borderParticles->getPositions();
	const vector<Real>& borderVolume = borderParticles->getVolumes();
	const Real c = fluidParticles->getCompactSupport();
	const Real h = fluidParticles->getSmoothingLength();
	const Real flMass = fluidParticles->getMass();
	const Real brRestDensity = borderParticles->getRestDensity();
	const Real flRestDensity = fluidParticles->getRestDensity();

	//compute normals
	vector<Vector3R> normals(fluidParticles->size(), Vector3R(0,0,0));
	#pragma omp parallel for schedule(guided, 100)
	for(int i = 0; i < fluidParticles->size(); i++){
		for(unsigned int j : neighbors[i][0]){
			normals[i] += kernelGradFunction(positions[i], positions[j], h) / 
							densities[j];
		}
		normals[i] = normals[i] * c * flMass;
	}
	//compute forces
	#pragma omp parallel for schedule(guided, 100)
	for(int i = 0; i < fluidParticles->size(); i++){
		Vector3R acc_curv = Vector3R(0,0,0);
		Vector3R acc_coh = Vector3R(0,0,0);
		//cohesion and curvature forces
		for(unsigned int j : neighbors[i][0]){
			Real Kij = 2 * flRestDensity / (densities[i] + densities[j]);
			acc_curv += Kij * (normals[i] - normals[j]);
			acc_coh += Kij * kernelCohetion(positions[i], positions[j], c) *
						(positions[i] - positions[j]).normalized();
		}
		acc_coh = flMass * flMass * acc_coh;
		//adhesion force
		Vector3R acc_adhesion = Vector3R(0,0,0);
		for(unsigned int j : neighbors[i][1]){
			acc_adhesion += borderVolume[j] * brRestDensity *
							kernelAdhesion(positions[i], borderPositions[j], c) * 
							(positions[i] - borderPositions[j]).normalized();
		}
		accelerations[i] -= lambda * (acc_curv + acc_coh);
		accelerations[i] -= betha * acc_adhesion;
	}
}

vector<Real> learnSPH::compute_curvature(const FluidSystem *fluidParticles)
{
	const Real c = fluidParticles->getCompactSupport();
	const Real h = fluidParticles->getSmoothingLength();
	const Real flMass = fluidParticles->getMass();
	const vector<Real>& densities = fluidParticles->getDensities();
	const vector<Vector3R>& positions = fluidParticles->getPositions();
	const vector<vector<vector<unsigned int>>>& neighbors = fluidParticles->getNeighbors();

	//compute normals
	vector<Vector3R> normals(fluidParticles->size(), Vector3R(0,0,0));
	#pragma omp parallel for schedule(guided, 100)
	for(size_t i = 0; i < fluidParticles->size(); i++){
		for(unsigned int j : neighbors[i][0]){
			normals[i] += kernelGradFunction(positions[i], positions[j], h) /
							densities[j];
		}
		normals[i] = normals[i] * c * flMass;
	}
	std::vector<Real> curvature(fluidParticles->size(), 0);
	#pragma omp parallel for schedule(guided, 100)
	for(size_t i = 0; i < fluidParticles->size(); i++){
		for(auto nb : neighbors[i][0])
			curvature[i] += (1 - normals[i].dot(normals[nb])) * std::max(0., 1 - std::fabs((positions[i] - positions[nb]).norm()) / h);
	}

	return curvature;

}
