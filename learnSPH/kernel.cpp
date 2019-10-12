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
	assert(q >= 0.0);
	constexpr double sigma = 3. /(2. * PI);

	if (q < 1.0) {
		return sigma * (-2*q + 1.5*pow2(q));
	}
	else if (q < 2.0) {
		return sigma * (-3/6. * pow2(2-q));
	}
	else {
		return 0;
	}
	assert(0 && "should never reach here");

	return 0;
}

double learnSPH::kernel::kernelFunction(const double q)
{
	assert(q >= 0.0);
	//smoosing length wath empirically found
	return 1./pow3(smooth_length) * cubicFunction(q);
}

double learnSPH::kernel::kernelFunction(const Vector3d& x1, const Vector3d& x2)
{
	return kernelFunction((x1-x2).norm());
}


Vector3d learnSPH::kernel::kernelGradFunction(const Vector3d& x1, const Vector3d& x2)
{
	//smoosing length wath empirically found
	Vector3d distVec = x2-x1;
	double q = distVec.norm();
	return 1./pow3(smooth_length) * cubicGradFunction(q) * distVec/q;
}
