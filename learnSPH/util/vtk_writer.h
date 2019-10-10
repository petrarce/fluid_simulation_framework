#pragma once
#include <vector>
#include <fstream>
#include <iostream>

#include <Eigen/Dense>

namespace learnSPH
{
	void saveParticlesToVTK(std::string path, const std::vector<Eigen::Vector3d>& particles, const std::vector<double>& particle_scalar_data, const std::vector<Eigen::Vector3d>& particle_vector_data);

	template<class T>
	void swapBytesInplace(T* arr, const int size)
	{
		constexpr unsigned int N = sizeof(T);
		constexpr unsigned int HALF_N = N / 2;
		char* char_arr = reinterpret_cast<char*>(arr);
		for (int e = 0; e < size; e++) {
			for (int w = 0; w < HALF_N; w++) {
				std::swap(char_arr[e * N + w], char_arr[(e + 1) * N - 1 - w]);
			}
		}
	}
}


