#include <solver.h>
#include <kernel.h>
opcode Solver::calculate_dencities(NormalPartDataSet& fluidParticles,
	const BorderPartDataSet& borderParticles,
	const vector<vector<vector<unsigned int>>>& fluidParticleNeighbours)
{
	//fluidParticleNeighbours should contain set of neighbors for each particle of the fluid
	assert(fluidParticleNeighbours.size() == fluidParticles.getNumberOfParticles());
	//for each particle there should be two sets of neighbors: 
	//	fluid neighbors set and border neighbor set
	assert(fluidParticleNeighbours[0].size() == 2);

	vector<Real>& fluidParticlesDensities = fluidParticles.getParticleDencities();
	const Vector3R* fluidParticlesPositions = fluidParticles.getParticlePositionsData();
	const Vector3R* borderParticlePositions = borderParticles.getParticlePositionsData();
	const Real* borderParticlesVolumes = borderParticles.getParticleVolumeData();
	const Real smoothingLength = fluidParticles.getParticleDiameter()*compactSupportFactor;


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
										smoothingLength);
		}
		borderDensity *= borderParticles.getRestDensity();

		fluidParticlesDensities[i] = fluidDensity + borderDensity;
	}
	return STATUS_OK;
}
	
