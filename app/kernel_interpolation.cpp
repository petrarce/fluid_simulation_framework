#include <types.hpp>
#include <iostream>
#include <kernel.h>
#include <vector>
#include <cmath>
#include <CompactNSearch.h>
#include <Eigen/Dense>

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
	vector<array<Real, 3>> pset2;


	int npoints = stoi(argv[1]);
	Real sml = stod(argv[2]);//smoothing length

	srand(1);
#ifdef _OPENMP
	#pragma omp parallel
	#pragma omp single
		pr_dbg("openmp threads enabled: %d", omp_get_num_threads());
	int max_num_threads = omp_get_max_threads();
#endif

	pset1.reserve(npoints);
	pset1.resize(npoints);
	//resetve 1 more point for further neighbourhood search on points of previous set
	pset2.reserve(npoints + 1);
	pset2.resize(npoints + 1);
	for(int i = 0; i < npoints; i++){
		Real x1 = getRand(48, 50);
		Real y1 = getRand(48, 50);
		Real z1 = getRand(48, 50);
		Real x2 = getRand(48, 50);
		Real y2 = getRand(48, 50);
		Real z2 = getRand(48, 50);

		pset1[i] = {x1,y1,z1};
		pset2[i] = {x2,y2,z2};
	}

	vector<Real> scalar_set1;
	vector<Real> scalar_set2;
	vector<Real> estimated_scalar_set1;
	scalar_set1.reserve(npoints);
	scalar_set1.resize(npoints);
	scalar_set2.reserve(npoints);
	scalar_set2.resize(npoints);
	estimated_scalar_set1.reserve(npoints);
	estimated_scalar_set1.resize(npoints);

	constexpr int sched_val = 64/sizeof(array<Real, 3>) + 1;


	#pragma omp parallel for schedule(static, sched_val) num_threads(max_num_threads)
	for(int i = 0; i < npoints; i++){
		scalar_set1[i] = magn(pset1[i], {0,0,0});
		scalar_set2[i] = magn(pset2[i], {0,0,0});
	}

	NeighborhoodSearch nbs(sml/2.);
	int psetid = nbs.add_point_set(pset2.front().data(), npoints + 1, true, true);
	for(int i = 0; i < npoints; i++){
		pset2.back() = pset1[i];
		vector<vector<unsigned int>> neighbours;
		nbs.update_point_sets();
		nbs.find_neighbors(psetid, pset2.size() - 1, neighbours);
		assert(neighbours.size() == 1);
		estimated_scalar_set1[i] = 0;
		for(int j = 0; j < neighbours[0].size(); j++){
			Real kernel_val = kernelFunction(
					Vector3d(pset1[i][0], pset1[i][1], pset1[i][2]),
					Vector3d(	pset2[neighbours[0][j]][0], 
								pset2[neighbours[0][j]][1], 
								pset2[neighbours[0][j]][2]),
					sml);
			estimated_scalar_set1[i] = estimated_scalar_set1[i] + scalar_set2[j] * kernel_val;
		}
	}

	printf("\"smoothing length: %f\",\"real magnitude \",\"interpolated magnitude\",\"error\"\n", sml);
	for(int i = 0; i < npoints; i++){
		printf("\"\", \"%.05f\", \"%.05f\", \"%.05f\"\n",
			scalar_set1[i], estimated_scalar_set1[i], scalar_set1[i] - estimated_scalar_set1[i]);
	}





	return 0;
}