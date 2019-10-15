#pragma once
#include <cassert>
#include <Eigen/Dense>

using namespace Eigen;

namespace learnSPH
{
	namespace kernel
	{
		constexpr double PI = 3.14159265358979323846;
		constexpr double smooth_length = 0.2;
		constexpr double particle_size = 0.05; //meters
		static double pow2(const double a) { return a * a; }
		static double pow3(const double a) { return a * a * a; }

		inline double cubicFunction(const double q);
		inline double cubicGradFunction(const double q);
		double 		kernelFunction(const Vector3d& x1, const Vector3d& x2, const double h);
		Vector3d 	kernelGradFunction(const Vector3d& x1, const Vector3d& x2, const double h);

	};
};