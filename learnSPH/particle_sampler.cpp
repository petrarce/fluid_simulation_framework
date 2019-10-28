#include <data_set.h>
#include <particle_sampler.h>

using namespace std;
using namespace learnSPH;

ParticleDataSet* ParticleSampler::sample_normal_particles(const Vector3R& upper_corner,
	 											const Vector3R& lover_corner,
	 											const Real rest_dencity,
	 											const Real sampling_distance)
{
	
	Vector3R distVector = upper_corner - lover_corner;
	size_t num_of_part_x_direction = abs(distVector[0]/sampling_distance);
	size_t num_of_part_y_direction = abs(distVector[1]/sampling_distance);
	size_t num_of_part_z_direction = abs(distVector[2]/sampling_distance);

	Real delX = sampling_distance * distVector[0]/fabs(distVector[0]);
	Real delY = sampling_distance * distVector[1]/fabs(distVector[1]);
	Real delZ = sampling_distance * distVector[2]/fabs(distVector[2]);

	size_t totalNumOfPrticles = num_of_part_x_direction * 
										num_of_part_y_direction*
										num_of_part_z_direction;

	vector<PositionVector> particlePositions;
	vector<VelocVector> particleVelocities;
	vector<Real> particleDensities;

	particlePositions.resize(totalNumOfPrticles);
	particleDensities.resize(totalNumOfPrticles);
	particleVelocities.resize(totalNumOfPrticles);

	Real posX = lover_corner[0];
	#pragma omp parallel for schedule(static) firstprivate(posX)
	for(int i = 0; i < num_of_part_x_direction; i++){
		posX = lover_corner[0] + i * delX;
		Real posY = lover_corner[1];
		for(int j = 0; j < num_of_part_y_direction; j++){

			Real posZ = lover_corner[2];
			for(int k = 0; k < num_of_part_z_direction; k++, posZ += delZ){
				size_t index = i*num_of_part_y_direction*num_of_part_z_direction + 
								j*num_of_part_z_direction + k;
				assert(index < totalNumOfPrticles && index >= 0);
				particlePositions[index] = {posX, posY, posZ};
			}
			posY += delY;
		}
	}

	NormalPartDataSet* normParticles = new NormalPartDataSet(particlePositions, 
																particleVelocities,
																particleDensities, 
																sampling_distance);
	return normParticles;

}
ParticleDataSet* ParticleSampler::sample_border_particles(const Vector3R& corner_a, 
											const Vector3R& corner_b,
											const Vector3R& corner_c){ return NULL;};
