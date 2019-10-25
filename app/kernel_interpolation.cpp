#include <types.hpp>
#include <iostream>
#include <kernel.h>
#include <vector>
#include <cmath>
#include <CompactNSearch.h>
#include <Eigen/Dense>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace learnSPH::kernel;
using namespace CompactNSearch;
using namespace Eigen;


static inline double magn(const array<Real, 3>& lhs, const array<Real, 3>& rhs)
{
	return sqrt(pow2(lhs[0]-rhs[0]) + pow2(lhs[1]-rhs[1]) + pow2(lhs[2]-rhs[2]));
}

int main(int argc, char** argv){

	vector<array<Real, 3>> pset1;


	int npoints = stoi(argv[1]);
	Real sml = stod(argv[2]);//smoothing length
	double delta_step = 3./(int(cbrt(npoints)) - 1);
	int pts_per_dim = cbrt(npoints);
	npoints = pts_per_dim * pts_per_dim * pts_per_dim;
	srand(1);
#ifdef _OPENMP
	#pragma omp parallel
	#pragma omp single
		//pr_dbg("openmp threads enabled: %d", omp_get_num_threads());
	int max_num_threads = omp_get_max_threads();
#endif

	pset1.reserve(npoints);
	pset1.resize(npoints);
	//resetve 1 more point for further neighbourhood search on points of previous set
	for(int i = 0; i < pts_per_dim; i++){
		for(int j = 0; j < pts_per_dim; j++){
			for(int k = 0; k < pts_per_dim; k++){
				Real x = i*delta_step - 1.5;
				Real y = j*delta_step - 1.5;
				Real z = k*delta_step - 1.5;
				int index = i * pts_per_dim* pts_per_dim + j * pts_per_dim + k;
				pset1[index] = {x,y,z};
			}

		}
	}

	vector<Real> scalar_set1;
	vector<Real> estimated_scalar_set1;
	scalar_set1.reserve(npoints);
	scalar_set1.resize(npoints);
	estimated_scalar_set1.reserve(npoints);
	estimated_scalar_set1.resize(npoints);

	for(int i = 0; i < npoints; i++){
		scalar_set1[i] = magn(pset1[i], {0,0,0});
	}

	NeighborhoodSearch nbs(sml);
	int psetid = nbs.add_point_set(pset1.front().data(), pset1.size(), true, true);
	vector<vector<unsigned int>> neighbours;
	nbs.update_point_sets();

	auto t0 = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < npoints; i++){
		nbs.find_neighbors(psetid, i, neighbours);

		assert(neighbours.size() == 1);

		Real total_weight = 0;
		Real kernel_weight = 0;
		estimated_scalar_set1[i] = 0;
		for(unsigned int j : neighbours[0]){
			kernel_weight = kernelFunction(Vector3d(pset1[i][0], pset1[i][1], pset1[i][2]),
											Vector3d(pset1[j][0], pset1[j][1], pset1[j][2]),
											sml);
			estimated_scalar_set1[i] += kernel_weight * scalar_set1[j];
			total_weight += kernel_weight;
		}

		if(total_weight != 0.0){
			estimated_scalar_set1[i] = estimated_scalar_set1[i] / total_weight;
		}

	}
	auto t1 = std::chrono::high_resolution_clock::now();

	Real squared_error;
	for(int i = 0; i < estimated_scalar_set1.size(); i++){
		squared_error += pow2(estimated_scalar_set1[i] - scalar_set1[i]);
	}

	squared_error = sqrt(squared_error)/estimated_scalar_set1.size();
	printf("%d %f %f %ld\n", npoints, sml, squared_error, 
		std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
	return 0;
}