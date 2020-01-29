#pragma once
#include <cassert>
#include <Eigen/Dense>
#include <types.hpp>

using namespace Eigen;

namespace learnSPH
{
	namespace kernel
	{
		constexpr double PI = 3.1416;

		static double pow2(const double a) { return a * a; }
		static double pow3(const double a) { return a * a * a; }

		inline double cubicFunction(const double q);
		inline double cubicGradFunction(const double q);

		double kernelFunction(const Vector3d& x1, const Vector3d& x2, const double h);
		Vector3R kernelGradFunction(const Vector3d& x1, const Vector3d& x2, const double h);

		double kernelCohesion(const Vector3d& x1, const Vector3d& x2, double c);
		double kernelAdhesion(const Vector3d& x1, const Vector3d& x2, double c);
	};
};