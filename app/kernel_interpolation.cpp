#include <types.hpp>
#include <iostream>
#include <kernel.h>
#include <vector>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace learnSPH::kernel;

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
	pset2.reserve(npoints);
	pset2.resize(npoints);
	for(int i = 0; i < npoints; i++){
		Real x1 = getRand(-1, 1);
		Real y1 = getRand(-1, 1);
		Real z1 = getRand(-1, 1);
		Real x2 = getRand(-1, 1);
		Real y2 = getRand(-1, 1);
		Real z2 = getRand(-1, 1);

		pset1[i] = {x1,y1,z1};
		pset2[i] = {x2,y2,z2};
	}

	vector<Real> scalar_set1;
	vector<Real> scalar_set2;
	vector<Real> err;
	scalar_set1.reserve(npoints);
	scalar_set1.resize(npoints);
	scalar_set2.reserve(npoints);
	scalar_set2.resize(npoints);
	err.reserve(npoints);
	err.resize(npoints);
	constexpr int sched_val = 64/sizeof(array<Real, 3>) + 1;

	#pragma omp parallel for schedule(static, sched_val) num_threads(max_num_threads)
	for(int i = 0; i < npoints; i++){

		scalar_set1[i] = magn(pset1[i], pset2[i]);
		scalar_set2[i] = kernelFunction(Vector3d(pset1[i][0], pset1[i][1], pset1[i][2]),
											 Vector3d(pset2[i][0], pset2[i][1], pset2[i][2]), 
											 sml);
	}


	#pragma omp parallel for schedule(static, sched_val) num_threads(max_num_threads)
	for(int i = 0; i < npoints; i++){
		err[i] = scalar_set1[i] - scalar_set2[i];
	}

	printf("\"vector_set1\",\"vector_set2\",\"magnitude\",\"kernel_funktion\",\"error\"\n");
	for(int i = 0; i < npoints; i++){
		printf("\"[%.05f,%.05f, %.05f]\",\"[%.05f,%.05f, %.05f]\","
			"\"%.05f\", \"%.05f\", \"%.05f\"\n",
			pset1[i][0], pset1[i][1], pset1[i][2],
			pset2[i][0], pset2[i][1], pset2[i][2],
			scalar_set1[i], scalar_set2[i], err[i]);
	}





	return 0;
}