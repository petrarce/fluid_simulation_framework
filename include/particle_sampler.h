#pragma once
#include <data_set.h>

using namespace std;
using namespace learnSPH;

namespace learnSPH{
	class ParticleSampler
	{

	private:
		static opcode sample_border_points_in_triangle(const Vector3R& corner_a, 
	 												const Vector3R& corner_b,
	 												const Vector3R& corner_c,
	 												const Real samplingDistance,
	 												vector<Vector3R>& borderParticleSet);
		static opcode sample_border_points_in_box(const Vector3R& upperCorner,
													const Vector3R& lowerCorner,
	 												const Real samplingDistance,
	 												vector<Vector3R>& borderParticleSet);

	public:
		/*
		generate fluid particles inside a cube with upper_corner and lower_corner
			upperCorner 		- upper corner of the cube, in which particles will be generated
			lowerCorner 		- lower corner of the cube, in which particles will be generated
			restDensity 		- rest density for all particle (constant among each particles)
			samplingDistance 	- distance between centers of particles
		*/
		static ParticleDataSet* sample_normal_particles(const Vector3R& upperCorner,
												const Vector3R& lowerCorner,
												const Real restDensity,
												const Real samplingDistance);
	 	static ParticleDataSet* sample_border_triangle(const Vector3R& corner_a, 
	 												const Vector3R& corner_b,
	 												const Vector3R& corner_c,
													const Real particleDensities,
	 												const Real samplingDistance);
	 	static ParticleDataSet* sample_border_box(const Vector3R& upperCorner,
													const Vector3R& lowerCorner,
													const Real particleDensities,
		 											const Real samplingDistance);
	private:
		ParticleSampler(){};
		~ParticleSampler(){};
		const ParticleSampler& operator=(ParticleSampler& other){return *this;};
	};
};