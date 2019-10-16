#include "catch.hpp"
#include "../learnSPH/kernel.h"
#include <iostream>

using namespace learnSPH::kernel;
#define pr_dbg(msg, args...) fprintf(stdout, "[DBG (%s:%d)]" msg "\n", __FILE__, __LINE__, ##args)

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
		kernelFunction(x1, x2 + epsX, h) - kernelFunction(x1, x2 - epsX, h),
		kernelFunction(x1, x2 + epsY, h) - kernelFunction(x1, x2 - epsY, h),
		kernelFunction(x1, x2 + epsZ, h) - kernelFunction(x1, x2 - epsZ, h)
	);
	return 0.5 * (1./eps) * gradVect;
}

double getRand(const double minVal,const double maxVal){
	assert(maxVal - minVal >= 0.0);
	return minVal + rand()/RAND_MAX * (maxVal - minVal);
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
 	SECTION("Basic tests"){
 		REQUIRE(pow2(2) == 4);
 		REQUIRE(pow2(123.456) > 15241);
 		REQUIRE(pow2(123.456) < 15242);
 		REQUIRE(pow3(3) == 27);
 		REQUIRE(pow3(123.456) > 1881640);
 		REQUIRE(pow3(123.456) < 1881641);
 	}
 	srand(10);
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
 }