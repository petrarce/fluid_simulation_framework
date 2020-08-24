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

double learnSPH::kernel::kernelCohetion(const Vector3d& x1, const Vector3d& x2, const double c)
{
	assert(c > 0);
	double r = (x1 - x2).norm();
	double factor = 32.f / (PI * pow(c,9));
	if(r >= 0 && r <= c/2){
		return factor * 2 * pow3(c - r)*pow3(r) - pow(c,6)/64.0f;
	} else if(r > c/2 && r <= c){
		return factor * pow3(c - r)*pow3(r);
	} else{
		return 0;
	}
}

double learnSPH::kernel::kernelAdhesion(const Vector3d& x1, const Vector3d& x2, double c)
{
	assert(c > 0);
	double r = (x1-x2).norm();
	if(r <= c && r > 0.5 * c){
		return (0.007 / pow(c,3.25)) * pow(-4 * pow2(r)/c + 6 * r - 2 * c, 0.25);
	}
	return 0;
}

double learnSPH::kernel::kernelCubic(const Vector3d& x1, const Vector3d& x2, float R)
{
    assert(R > 0);
    double s = (x1 - x2).norm() / R;
    double v = (1 - s*2);
    return std::max(-1., std::min(1., v * v * v));

};
