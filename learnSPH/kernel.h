#pragma once
#include <cassert>
#include <Eigen/Dense>

namespace learnSPH
{
	namespace kernel
	{
		constexpr double PI = 3.14159265358979323846;
		static double pow2(const double a) { return a * a; }
		static double pow3(const double a) { return a * a * a; }

		inline double cubicFunction(const double q);
		inline double cubicGradFunction(const double q);
	};
};