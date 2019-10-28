#include <data_set.h>

using namespace std;
using namespace learnSPH;

namespace learnSPH{
	class ParticleSampler
	{
	private:
		ParticleSampler(){};
		~ParticleSampler(){};
		const ParticleSampler& operator=(){return *this;};

	public:
	 	ParticleDataSet* sample_normal_particles(const Vector3R& upper_corner
	 											const Vector3R& lover_corner,
	 											const Real rest_dencity);
	 	ParticleDataSet* sample_border_particles(const Vector3R& corner_a, 
	 												const Vector3R& corner_b,
	 												const Vector3R& corner_c);
	};
};