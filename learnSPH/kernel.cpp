#include "kernel.h"

double learnSPH::kernel::cubicFunction(const double q)
{
	assert(q >= 0.0);
	constexpr double sigma = 3.0 / (2.0 * PI);

	if (q < 1.0) {
		return sigma * (2.0 / 3.0 - pow2(q) + 0.5 * pow3(q));
	}
	else if (q < 2.0) {
		return sigma * (1.0 / 6.0 * pow3(2.0 - q));
	}
	else {
		return 0.0;
	}
}

double learnSPH::kernel::cubicGradFunction(const double q)
{
	assert(q >= 0.0);
	constexpr double sigma = 3.0 / (2.0 * PI);

	if (q < 1.0) {
		return sigma * (-2.0 * q + 1.5 * pow2(q));
	}
	else if (q < 2.0) {
		return sigma * (-3.0 / 6.0 * pow2(2.0 - q));
	}
	else {
		return 0.0;
	}
}

double learnSPH::kernel::kernelFunction(const Vector3d& x1, const Vector3d& x2, const double h)
{
	return 1.0 / pow3(h) * cubicFunction((x1 - x2).norm() / h);
}


Vector3d learnSPH::kernel::kernelGradFunction(const Vector3d& x1, const Vector3d& x2, const double h)
{
	return 1.0 / (pow2(h) * pow2(h)) * cubicGradFunction((x1 - x2).norm() / h) * (x1 - x2).normalized();
}

double learnSPH::kernel::kernelCohesion(const Vector3d& x1, const Vector3d& x2, double c)
{
	double r = (x1 - x2).norm();

	double factor = 32.0 / (PI * pow(c, 9));

	if (r <= c / 2.0) return factor * 2.0 * pow3(c - r) * pow3(r) - pow(c, 6) / 64.0;

	if (r > c / 2.0 && r <= c) return factor * pow3(c - r) * pow3(r);

	return 0.0;
}

double learnSPH::kernel::kernelAdhesion(const Vector3d& x1, const Vector3d& x2, double c)
{
	double r = (x1 - x2).norm();

	if (r <= c) return r - pow2(r);

	return 0.0;
}
