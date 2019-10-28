#include <types.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace learnSPH
{
	class ParticleDataSet {
	protected:
		Vector3R *ptr_to_particle_set;
		int num_particles;
		Real rest_density;

	public:
		enum ParticleType{
			BORDER = 0,
			NORMAL
		};
	public:

		virtual ParticleType get_type() = 0;
		Vector3R* data(){return ptr_to_particle_set;};
		size_t get_num_particles(){return num_particles;};

		ParticleDataSet(Vector3R *part_ptr, size_t num_part, Real rest_dens):
			ptr_to_particle_set(part_ptr),
			num_particles(num_part),
			rest_density(rest_dens){};
		virtual ~ParticleDataSet(){};
	};

	class BorderPartDataSet : ParticleDataSet{
		
	public:
		virtual ParticleType get_type(){return BORDER;};
		BorderPartDataSet(Vector3R *part_ptr, size_t num_part, Real rest_dens):
			ParticleDataSet(part_ptr, num_part, rest_dens){};
		~BorderPartDataSet(){};
	};

	class NormalPartDataSet : ParticleDataSet{
		Real *ptr_to_density_set;
	public:
		virtual ParticleType get_type(){return NORMAL;};
		Real* data_dencities(){return ptr_to_density_set;};
		NormalPartDataSet(Vector3R *part_ptr, Real *ptr_dens, size_t num_part, Real rest_dens):
			ParticleDataSet(part_ptr, num_part, rest_dens),
			ptr_to_density_set(ptr_dens){};
		~NormalPartDataSet(){};
	};
};