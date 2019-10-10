#include "kernel.h"

double learnSPH::kernel::cubicFunction(const double q)
{
	assert(q >= 0.0);
	constexpr double sigma = 1.0 / PI;

	if (q < 1.0) {
		return sigma * (0.25 * pow3(2.0 - q) - pow3(1.0 - q));
	}
	else if (q < 2.0) {
		// ...
	}
	else {
		// ...
	}
}

double learnSPH::kernel::cubicGradFunction(const double q)
{
	// ...
}

