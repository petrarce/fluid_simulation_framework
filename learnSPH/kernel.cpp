#include "kernel.h"

double learnSPH::kernel::cubicFunction(const double q)
{
	assert(q >= 0.0);
	constexpr double sigma = 3. /(2. * PI);

	if (q < 1.0) {
		return sigma * (2./3. - pow2(q) + 0.5*pow3(q));
	}
	else if (q < 2.0) {
		return sigma * (1/6. * pow3(2 - q));
	}
	else {
		return 0;
	}
	assert(0 && "should never reach here");
}

double learnSPH::kernel::cubicGradFunction(const double q)
{
	// ...
	return 0;
}

double learnSPH::kernel::kernelFunction(const double q)
{
	assert(q >= 0.0);
	//smoosing length wath empirically found
	constexpr double h = 2.1;
	return 1./pow3(h) * cubicFunction(q);
}
double learnSPH::kernel::kernelGradFunction(const double q)
{
	//...
	return 0;
}
