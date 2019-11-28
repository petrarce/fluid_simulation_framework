#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>

#include <types.hpp>
#include <marching_cubes.h>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

using namespace learnSPH;
using namespace std;

void load_vectors(const std::string &path, std::vector<Vector3R> &data)
{
	std::ifstream is(path, std::ios::binary);

	cereal::BinaryInputArchive bia(is);

	size_t n_elems;

	bia(n_elems);

	data.reserve(n_elems);

	for (auto i = 0; i < n_elems; i++) {

		Real x;
		Real y;
		Real z;

		bia(x, y, z);

		data.push_back(Vector3R(x, y, z));
	}
}

void load_scalars(const std::string &path, std::vector<Real> &data)
{
	std::ifstream is(path, std::ios::binary);

	cereal::BinaryInputArchive bia(is);

	size_t n_elems;

	bia(n_elems);

	data.reserve(n_elems);

	for (auto i = 0; i < n_elems; i++) {

		Real x;

		bia(x);

		data.push_back(x);
	}
}

int main(int argc, char** argv)
{
	std::cout << "Per frame rendering running" << std::endl;

	Vector3R lower_corner_box(stod(argv[0]),stod(argv[1]),stod(argv[2]));
    Vector3R upper_corner_box(stod(argv[3]),stod(argv[4]),stod(argv[5]));

    Real render_step = stod(argv[6]);
    Real sim_duration = stod(argv[7]);
    string sim_name = argv[8];

    Vector3R cubeResolution(stod(argv[9]), stod(argv[10]), stod(argv[11]));

    Real initValue = stod(argv[12]);

    MarchingCubes mcb(lower_corner_box, upper_corner_box, cubeResolution);

    unsigned int nsamples = int(sim_duration / render_step);

    for (unsigned int t = 0; t < nsamples; t++) {

    	cout << "\nframe [" << t << "] rendered" << endl;
    }
}