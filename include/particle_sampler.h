#pragma once
#include <data_set.h>

using namespace std;
using namespace learnSPH;

namespace learnSPH{
	class ParticleSampler
	{


	public:
		/*
		generate fluid particles inside a cube with upper_corner and lover_corner
			uppderCorner 		- upper corner of the cube, in which particles will be generated
			loverCorner 		- lover corner of the cube, in which particles will be generated
			restDensity 		- rest density for all particle (constant among each particles)
			samplingDistance 	- distance between centres of particles
		*/
		static ParticleDataSet* sample_normal_particles(const Vector3R& uppderCorner,
												const Vector3R& loverCorner,
												const Real restDensity,
												const Real samplingDistance);
	 	static ParticleDataSet* sample_border_particles(const Vector3R& corner_a, 
	 												const Vector3R& corner_b,
	 												const Vector3R& corner_c,
	 												const Real samplingDistance);
	private:
		ParticleSampler(){};
		~ParticleSampler(){};
		const ParticleSampler& operator=(ParticleSampler& other){return *this;};
	};
};