#pragma once
#include <data_set.h>

using namespace std;
using namespace learnSPH;

namespace learnSPH{
	class ParticleSampler
	{


	public:
	 	static ParticleDataSet* sample_normal_particles(const Vector3R& upper_corner,
	 											const Vector3R& lover_corner,
	 											const Real rest_dencity,
	 											const Real sampling_distance);
	 	static ParticleDataSet* sample_border_particles(const Vector3R& corner_a, 
	 												const Vector3R& corner_b,
	 												const Vector3R& corner_c);
	private:
		ParticleSampler(){};
		~ParticleSampler(){};
		const ParticleSampler& operator=(ParticleSampler& other){return *this;};
	};
};