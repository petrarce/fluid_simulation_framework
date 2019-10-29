#include <data_set.h>
#include <particle_sampler.h>

using namespace std;
using namespace learnSPH;

constexpr Real treshold = 1e-6;

ParticleDataSet* ParticleSampler::sample_normal_particles(const Vector3R& upperCorner,
	 											const Vector3R& lowerCorner,
	 											const Real restDensiti,
	 											const Real samplingDistance)
{
	
	Vector3R distVector = upperCorner - lowerCorner;
	size_t num_of_part_x_direction = abs(distVector[0]/samplingDistance) + 1;
	size_t num_of_part_y_direction = abs(distVector[1]/samplingDistance) + 1;
	size_t num_of_part_z_direction = abs(distVector[2]/samplingDistance) + 1;

	Real delX = samplingDistance * distVector[0]/fabs(distVector[0]);
	Real delY = samplingDistance * distVector[1]/fabs(distVector[1]);
	Real delZ = samplingDistance * distVector[2]/fabs(distVector[2]);

	size_t totalNumOfPrticles = num_of_part_x_direction * 
										num_of_part_y_direction*
										num_of_part_z_direction;

	vector<PositionVector> particlePositions;
	vector<VelocVector> particleVelocities;
	vector<Real> particleDensities;

	particlePositions.resize(totalNumOfPrticles);
	particleDensities.resize(totalNumOfPrticles);
	particleVelocities.resize(totalNumOfPrticles);

	Real posX = lowerCorner[0];
	#pragma omp parallel for schedule(static) firstprivate(posX)
	for(int i = 0; i < num_of_part_x_direction; i++){
		posX = lowerCorner[0] + i * delX;
		Real posY = lowerCorner[1];
		for(int j = 0; j < num_of_part_y_direction; j++){

			Real posZ = lowerCorner[2];
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
																restDensiti,
																samplingDistance);
	return normParticles;

}

static inline bool check_point_in_triangle(const Vector3R& corner_a, 
											const Vector3R& corner_b,
											const Vector3R& corner_c,
											const Vector3R& point_p)
{
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

opcode ParticleSampler::sample_border_points_in_triangle(const Vector3R& corner_a, 
												const Vector3R& corner_b,
												const Vector3R& corner_c,
												const Real samplingDistance,
												vector<Vector3R>& borderParticleSet)
{
	Vector3R u1 = corner_b - corner_a;
	Vector3R v1 = corner_c - corner_a;
	assert(fabs(u1.normalized().dot(v1.normalized())) - 1 < treshold 
		&& "all points of the triangle are on the same line");
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

	borderParticleSet.clear();
	borderParticleSet.reserve(maxi*maxj);
	for(int i = 0; i < maxi; i++){
		for(int j = 0; j < maxj; j++){
			Vector3R newPoint = corner_a + i*v + j*u;
			if(check_point_in_triangle(corner_a, corner_b, corner_c, newPoint)){
				borderParticleSet.push_back(newPoint);
			}
		}
	}
	return STATUS_OK;
}


ParticleDataSet* ParticleSampler::sample_border_triangle(const Vector3R& corner_a, 
											const Vector3R& corner_b,
											const Vector3R& corner_c,
											const Real particleDensities,
											const Real samplingDistance)
{ 

	vector<Vector3R> borderParticleSet;
	sample_border_points_in_triangle(corner_a, 
										corner_b, corner_c, 
										samplingDistance,
										borderParticleSet);

	BorderPartDataSet* particleSet = new BorderPartDataSet(borderParticleSet, 
															particleDensities,
															samplingDistance);
	return particleSet;
};

ParticleDataSet* ParticleSampler::sample_border_box(const Vector3R& uppderCorner,
											const Vector3R& loverCorner,
											const Real particleDensities,
											const Real samplingDistance)
{
	Real a = loverCorner(0), b = loverCorner(1), c = loverCorner(2);
	Real d = uppderCorner(0), e = uppderCorner(1), f = uppderCorner(2);
	Real boxHeight = fabs(c-f);
	Real boxLength = fabs(b-e);
	Real boxWidth = fabs(a-d);
	vector<Vector3R> boxBorderPoints;
	//resetve points precisely 
	unsigned int pointsPerBoxSurface = 
		static_cast<unsigned int>(
			2*(boxHeight*boxLength + boxHeight*boxWidth + boxWidth*boxLength) 
			* 1.1 
			/ samplingDistance
		);
	boxBorderPoints.reserve(pointsPerBoxSurface);

	vector<Vector3R> prticleTriangelSet;
	sample_border_points_in_triangle(Vector3R(a,b,c),
							Vector3R(a,e,c),
							Vector3R(d,e,c),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());
	sample_border_points_in_triangle(Vector3R(a, b,c),
							Vector3R(d, b,c),
							Vector3R(d, e,c),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());


	sample_border_points_in_triangle(Vector3R(d, b,c),
							Vector3R(d, b,f),
							Vector3R(d, e,f),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());
	sample_border_points_in_triangle(Vector3R(d, b,c),
							Vector3R(d, e,c),
							Vector3R(d, e,f),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());

	sample_border_points_in_triangle(Vector3R(a, e,c),
							Vector3R(d, e,f),
							Vector3R(d, e,c),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());
	sample_border_points_in_triangle(Vector3R(a, e,c),
							Vector3R(a, e,f),
							Vector3R(d, e,f),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());

	sample_border_points_in_triangle(Vector3R(a, b,c),
							Vector3R(d, b,c),
							Vector3R(d, b,f),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());
	sample_border_points_in_triangle(Vector3R(a, b, c),
							Vector3R(d, b, f),
							Vector3R(a, b, f),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());

	sample_border_points_in_triangle(Vector3R(a, b, f),
							Vector3R(a, e, f),
							Vector3R(d, e, f),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());
	sample_border_points_in_triangle(Vector3R(a, b, f),
							Vector3R(d, b, f),
							Vector3R(d, e, f),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());

	sample_border_points_in_triangle(Vector3R(a, b, c),
							Vector3R(a, b, f),
							Vector3R(a, e, f),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());
	sample_border_points_in_triangle(Vector3R(a, b, c),
							Vector3R(a, e, c),
							Vector3R(a, e, f),
							samplingDistance, 
							prticleTriangelSet);
	boxBorderPoints.insert(boxBorderPoints.end(), 
							prticleTriangelSet.begin(), 
							prticleTriangelSet.end());

	BorderPartDataSet* particleSet = new BorderPartDataSet(boxBorderPoints, 
															particleDensities,
															samplingDistance);
	return particleSet;
}
