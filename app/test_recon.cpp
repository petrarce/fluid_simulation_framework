#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>

#include <vtk_writer.h>
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
	assert(argc == 14);

	std::cout << "Per frame rendering running" << std::endl;

	Vector3R lower_corner_box(stod(argv[1]),stod(argv[2]),stod(argv[3]));
	Vector3R upper_corner_box(stod(argv[4]),stod(argv[5]),stod(argv[6]));

	Real render_step = stod(argv[7]);
	Real sim_duration = stod(argv[8]);
	string sim_name = argv[9];

	Vector3R cubeResolution(stod(argv[10]), stod(argv[11]), stod(argv[12]));

	Real initValue = stod(argv[13]);

	MarchingCubes mcb(lower_corner_box, upper_corner_box, cubeResolution);

	unsigned int nsamples = int(sim_duration / render_step);

	for (unsigned int t = 0; t < nsamples; t++) {

		vector<Real> params;
		vector<Vector3R> positions;
		vector<Real> densities;

		string filename = "res/assignment3/" + sim_name + "_params_" + std::to_string(t) + ".cereal";

		load_scalars(filename, params);

		filename = "res/assignment3/" + sim_name + "_positions_" + std::to_string(t) + ".cereal";

		load_vectors(filename, positions);

		filename = "res/assignment3/" + sim_name + "_densities_" + std::to_string(t) + ".cereal";

		load_scalars(filename, densities);

		mcb.setObject(new Fluid(params, positions, densities, initValue, lower_corner_box, upper_corner_box, cubeResolution));

        vector<Vector3R> triangle_mesh;

        mcb.getTriangleMesh(triangle_mesh);

        vector<array<int, 3>> triangles;

        for(int i = 0; i < triangle_mesh.size(); i += 3) triangles.push_back({i, i + 1, i + 2});

        std::string surface_filename = "res/assignment3/" + sim_name + "_surface" + std::to_string(t) + ".vtk";

        learnSPH::saveTriMeshToVTK(surface_filename, triangle_mesh, triangles);

		cout << "\nframe [" << t << "] rendered" << endl;
	}
}
