#include "NBsearch.hpp"
#include <types.hpp>
#include <cmath>

NBS::NBS(Real r):
	cs(r)
{}

NBS::~NBS()
{

}
	
opcode NBS::set_compack_support(Real cs)
{
	assert(cs > 0);
	this->cs = cs;
	return STATUS_OK;
}


typedef struct {
	Real x;
	Real y;
	Real z;
	double square_norm(){
		return x*x + y*y + z*z;
	}
} point_t;
const point_t operator-(const point_t &lhs,const point_t &rhs)
{
	point_t res = {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
	return res;

}
opcode NBS::bf_find_neighbours(Real const* data, const size_t n_particles,
		vector<vector<unsigned int>> &neighbor_list)
{
	if(!data || !n_particles){
		return STATUS_NOK;
	}

	neighbor_list.clear();
	neighbor_list.resize(n_particles);
	size_t estimated_neighbours = sqrt(n_particles);
	for(int i = 0; i < n_particles; i++){
		neighbor_list[i].reserve(estimated_neighbours);
		point_t* particle =(point_t*)(data) + i; 
		for(int j = 0; j < n_particles; j++){
			if(j != i){
				point_t* neighb_particle = (point_t*)(data) + j;
				point_t dist_vec = *particle - *neighb_particle;
				if(dist_vec.square_norm() > this->cs*this->cs){
					continue;
				}
				neighbor_list[i].push_back(j);
			}
		}
	}
	return STATUS_OK;
}