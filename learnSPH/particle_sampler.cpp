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
	size_t num_of_part_x_direction = abs(distVector[0]/sampling_distance) + 1;
	size_t num_of_part_y_direction = abs(distVector[1]/sampling_distance) + 1;
	size_t num_of_part_z_direction = abs(distVector[2]/sampling_distance) + 1;

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

static inline bool check_point_in_triangle(const Vector3R& corner_a, 
											const Vector3R& corner_b,
											const Vector3R& corner_c,
											const Vector3R& point_p)
{
	constexpr Real treshold = 1e-6;
	Vector3R sideAB = corner_b - corner_a;
	Vector3R sideAC = corner_c - corner_a;
	Vector3R sidePA = corner_a - point_p;
	Vector3R sidePB = corner_b - point_p;
	Vector3R sidePC = corner_c - point_p;

	Real triangleField = sideAB.cross(sideAC).norm();

	Real fld1 = sidePA.cross(sidePB).norm();
	Real fld2 = sidePB.cross(sidePC).norm();
	Real fld3 = sidePC.cross(sidePA).norm();

	if((fld1+fld2+fld3) - triangleField < treshold){
		return true;
	}

	return false;
}

ParticleDataSet* ParticleSampler::sample_border_particles(const Vector3R& corner_a, 
											const Vector3R& corner_b,
											const Vector3R& corner_c,
											const Real samplingDistance)
{ 
	Vector3R u1 = corner_b - corner_a;
	Vector3R v1 = corner_c - corner_a;
	//in case corner a is obuse angle - svitch directions
	if(u1.dot(v1) < 0){
		u1 = corner_a - corner_c;
		v1 = corner_b - corner_c;
	}
	Vector3R u = u1;
	Vector3R v = v1;

	v = u.cross(v);
	v = v.cross(u);
	u = u / u.norm();
	v = v / v.norm();

	Real proj_v_v1 = v1.dot(v);
	Real proj_u1_u = u1.dot(u);

	size_t maxi = (proj_v_v1 / samplingDistance) + 1;
	size_t maxj = (proj_u1_u / samplingDistance) + 1;

	v = v * samplingDistance;
	u = u * samplingDistance;

	vector<Vector3R> borderParticleSet;
	borderParticleSet.reserve(maxi*maxj);
	for(int i = 0; i < maxi; i++){
		for(int j = 0; j < maxj; j++){
			Vector3R newPoint = corner_a + i*v + j*u;
			if(check_point_in_triangle(corner_a, corner_b, corner_c, newPoint)){
				borderParticleSet.push_back(newPoint);
			}
		}
	}


	BorderPartDataSet* particleSet = new BorderPartDataSet(borderParticleSet, 0);
	return particleSet;
};
