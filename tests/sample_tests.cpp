#include "catch.hpp"
#include "../learnSPH/kernel.h"

using namespace learnSPH::kernel;

struct Foo
{
    bool is_bar() const
    {
        return true;
    }
};

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
			printf("%f\n", val);
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
 }