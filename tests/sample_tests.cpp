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

Vector3d kernelEstimatedGrad(const Vector3d& x1, const Vector3d& x2){
	constexpr double eps = 1./1000000;
	Vector3d epsX = Vector3d(eps, 0, 0);
	Vector3d epsY = Vector3d(0, eps, 0);
	Vector3d epsZ = Vector3d(0, 0, eps);

	Vector3d diffVec = x1 - x2;
	Vector3d gradVect = Vector3d(
		kernelFunction(x1, x2 + epsX) - kernelFunction(x1, x2 - epsX),
		kernelFunction(x1, x2 + epsY) - kernelFunction(x1, x2 - epsY),
		kernelFunction(x1, x2 + epsZ) - kernelFunction(x1, x2 - epsZ)
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
			double val = (double(rand())/RAND_MAX) * (RAND_MAX - 3) + 2.1;
 			REQUIRE(kernelFunction(val) == 0);
 		}

 		//Non negative property
 		for(int i = 0; i < 100; i++){
 			REQUIRE(kernelFunction((double(rand())/RAND_MAX) * 2) >= 0);
 		}

 		//Unity property
 		double sum = 0;
 		double vq = -3;
 		for(int i = 0; i < 100; i++, vq+= 0.06){
 			double kfval = kernelFunction(abs(vq));
 			sum += kfval;
 		}
 		REQUIRE(sum < 1);

 	}

 	SECTION("KF gradient"){
 		double err = 0;
 		for(int i = 0; i < 100; i++){
 			Vector3d vec1 = Vector3d().Random();
 			if(vec1.norm() > 1){
 				vec1 = vec1/getRand(vec1.norm(), vec1.norm()*2);
 			}
  			Vector3d vec2 = Vector3d().Random();
 			if(vec2.norm() > 1){
 				vec2 = vec1/getRand(vec2.norm(), vec2.norm()*2);
 			}
 			REQUIRE(vec1.norm() <= 1.01);
 			REQUIRE(vec2.norm() <= 1.01);
 			Vector3d errVec = kernelGradFunction(vec1, vec2) - kernelEstimatedGrad(vec1, vec2);
 			err += errVec.norm();
 		}
 		err /= 100;
 		REQUIRE(err < 0.05);
 	}
 }