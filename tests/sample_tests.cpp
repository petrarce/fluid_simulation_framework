#include "catch.hpp"
#include <kernel.h>
#include <iostream>
#include <types.hpp>
#include <CompactNSearch.h>
#include <NBsearch.hpp>
#include <Config.h>
#include <random>
#include <algorithm>
#include <chrono>

using namespace learnSPH::kernel;
using namespace CompactNSearch;

struct Foo
{
    bool is_bar() const
    {
        return true;
    }
};

Vector3d kernelEstimatedGrad(const Vector3d& x1, const Vector3d& x2, const double h){
	constexpr double eps = 1./1000000;
	Vector3d epsX = Vector3d(eps, 0, 0);
	Vector3d epsY = Vector3d(0, eps, 0);
	Vector3d epsZ = Vector3d(0, 0, eps);

	Vector3d diffVec = x1 - x2;
	Vector3d gradVect = Vector3d(
		kernelFunction(x1 + epsX, x2, h) - kernelFunction(x1 - epsX, x2, h),
		kernelFunction(x1 + epsY, x2, h) - kernelFunction(x1 - epsY, x2, h),
		kernelFunction(x1 + epsZ, x2, h) - kernelFunction(x1 - epsZ, x2, h)
	);
	return 0.5 * (1./eps) * gradVect;
}

double getRand(const double minVal,const double maxVal){
	assert(maxVal - minVal >= 0.0);
	return minVal + double(rand())/RAND_MAX * (maxVal - minVal);
}

// Check out https://github.com/catchorg/Catch2 for more information about how to use Catch2
TEST_CASE( "Foo is always Bar", "[Foobar]" )
{
    Foo foo;

    REQUIRE(foo.is_bar());
    CHECK(foo.is_bar());
}
 TEST_CASE( "Test kernel function", "[Kernel]")
 {
  	srand(10);
	SECTION("Basic tests"){
 		REQUIRE(pow2(2) == 4);
 		REQUIRE(pow2(123.456) > 15241);
 		REQUIRE(pow2(123.456) < 15242);
 		REQUIRE(pow3(3) == 27);
 		REQUIRE(pow3(123.456) > 1881640);
 		REQUIRE(pow3(123.456) < 1881641);
 	}
 	SECTION("KF tests"){
 		//Compactness property
		for(int i = 0; i < 100; i++){
			Vector3d vec1 = Vector3d().Zero();
			Vector3d vec2 = Vector3d().Ones() + Vector3d().Random();
			vec2 = getRand(2., RAND_MAX) * (vec2 / vec2.norm());
 			REQUIRE(kernelFunction(vec1, vec2, smooth_length) == 0);
 		}

 		//Non negative property
 		for(int i = 0; i < 100; i++){
			Vector3d vec1 = Vector3d().Zero();
			Vector3d vec2 = Vector3d().Ones() + Vector3d().Random();
			vec2 = getRand(0., 2.) * (vec2 / vec2.norm());
 			REQUIRE(kernelFunction(vec1, vec2, smooth_length) >= 0);
 		}

 		//Unity property
 		double sum = 0;
 		Vector3d vec1 = Vector3d().Zero();
 		Vector3d vec2 = -2 * Vector3d().Ones();
 		for(int i = 0; i < 100; i++){
 			Vector3d vec2_1 = vec2;
 			for(int j = 0; j < 100; j++){
  			Vector3d vec2_2 = vec2_1;
				for(int k = 0; k < 100; k++){
 					sum += kernelFunction(vec1, vec2_2, smooth_length) * pow3(particle_size);
 					vec2_2 += Vector3d(0, 0, particle_size);
 				}
 				vec2_1 += Vector3d(0, particle_size, 0);
 			}
 			vec2 += Vector3d(particle_size,0,0);
 		}
 		REQUIRE(sum < 1.01);
 	}
 	SECTION("KF gradient"){
 		double err = 0;
 		for(int i = 0; i < 100; i++){
 			Vector3d vec1 = Vector3d().Random();
 			if(vec1.norm() > 1){
 				vec1 = vec1/getRand(vec1.norm(), vec1.norm()*1.99);
 			}
  			Vector3d vec2 = Vector3d().Random();
 			if(vec2.norm() > 1){
 				vec2 = vec1/getRand(vec2.norm(), vec2.norm()*1.99);
 			}
 			REQUIRE((vec1 - vec2).norm() <= 2.);
 			Vector3d errVec = kernelGradFunction(vec1, vec2, smooth_length) - 
 				kernelEstimatedGrad(vec1, vec2, smooth_length);
 			err += errVec.norm();
 		}
 		err /= 100;
 		REQUIRE(err < 1./1000000);
 	}
 	SECTION("Neighbourhood search"){
 		vector<array<Real, 3>> points;
 		Real radius = 1;

 		//define
 		points.reserve(20);
 		for(int i = 0; i < 20; i ++){
			Real x = getRand(-1,1);
			Real y = getRand(-1,1);
			Real z = getRand(-1,1);
			points.push_back({x,y,z});
 		}
 		random_shuffle(points.begin(), points.end());
 		NBS custom_nbs(radius);
 		vector<vector<unsigned int>> neighbours;

 		for(int i = 0; i < neighbours.size(); i++){
 			for(int j = 0; j < neighbours[i].size(); j++){
 				size_t nb_ind = neighbours[i][j];
 				bool i_is_in_j = (neighbours[nb_ind].size())?false:true;
 				for(int k = 0; k < neighbours[nb_ind].size(); k++){
 					if(neighbours[nb_ind][k] == i){
 						i_is_in_j = true;
 						break;
 					}
 				}
 				REQUIRE(i_is_in_j == true);
 			}
 		}


#ifndef NPOINTS
#define NPOINTS 1000
#endif

		points.clear();
 		points.reserve(NPOINTS);
 		for(int i = 0; i < NPOINTS; i ++){
			Real x = getRand(-1,1);
			Real y = getRand(-1,1);
			Real z = getRand(-1,1);
			points.push_back({x,y,z});
 		}
 		custom_nbs.set_compack_support(2*smooth_length);
  		auto t0 = std::chrono::high_resolution_clock::now();
 		custom_nbs.bf_find_neighbours(points.front().data(), points.size(), neighbours);
 		auto t1 = std::chrono::high_resolution_clock::now();
		
		fprintf(stderr, "custom_nbs takes %ld ms on %d particles\n", 
			std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(), NPOINTS);

		
		NeighborhoodSearch nsearch(2*smooth_length, true);
		nsearch.add_point_set(points.front().data(), points.size(), true, true);
		nsearch.z_sort();
		for (auto i = 0u; i < nsearch.n_point_sets(); ++i)
		{
			auto const& d = nsearch.point_set(i);
			d.sort_field(points.data());

		}
		t0 = std::chrono::high_resolution_clock::now();
		nsearch.find_neighbors();
		fprintf(stderr, "CompactNSearch takes %ld ms on %d particles\n", 
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - t0).count(), NPOINTS);

 	}
 }