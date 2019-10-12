#pragma once
#include <cassert>
#include <Eigen/Dense>

using namespace Eigen;

namespace learnSPH
{
	namespace kernel
	{
		constexpr double PI = 3.14159265358979323846;
		constexpr double smooth_length = 0.785;
		static double pow2(const double a) { return a * a; }
		static double pow3(const double a) { return a * a * a; }

		inline double cubicFunction(const double q);
		inline double cubicGradFunction(const double q);
		double 		kernelFunction(const double q);
		double 		kernelFunction(const Vector3d& x1, const Vector3d& x2);
		Vector3d 	kernelGradFunction(const Vector3d& x1, const Vector3d& x2);

	};
};