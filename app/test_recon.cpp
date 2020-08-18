#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>
#include <experimental/filesystem>
#include <math.h>

#include <learnSPH/core/vtk_writer.h>
#include <types.hpp>
#include <learnSPH/surf_reconstr/marching_cubes.h>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

using namespace learnSPH;
using namespace std;

static void load_vectors(const std::string &path, std::vector<Vector3R> &data)
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

static void load_scalars(const std::string &path, std::vector<Real> &data)
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

static std::vector<std::string> filterPaths(const std::string& path, const std::string& pathPattern)
{
	std::vector<std::string> filteredPaths;
	std::regex pathRegex(pathPattern, std::regex::grep);
	for(const auto& p : std::experimental::filesystem::directory_iterator(path))
	{
		if(std::regex_search(p.path().string(), pathRegex))
			filteredPaths.push_back(p.path().string());
	}
	std::sort(filteredPaths.begin(), filteredPaths.end());
	return filteredPaths;
}

int main(int argc, char** argv)
{
	assert(argc == 13);

	std::cout << "Per frame rendering running" << std::endl;

	Vector3R lowerCorner(stod(argv[1]),stod(argv[2]),stod(argv[3]));
	Vector3R upperCorner(stod(argv[4]),stod(argv[5]),stod(argv[6]));

	string sim_name = argv[7];

	Vector3R cubeResol(stod(argv[8]), stod(argv[9]), stod(argv[10]));

	Real initValue = stod(argv[11]);
	std::string fileDir(argv[12]);

	std::vector<std::string> paramFiles = filterPaths(fileDir, sim_name + "_params_[0-9]*.cereal");
	std::vector<std::string> positionFiles = filterPaths(fileDir, sim_name + "_positions_[0-9]*.cereal");
	std::vector<std::string> densitiesFiles = filterPaths(fileDir, sim_name + ".*_densities_[0-9]*.cereal");
	assert(paramFiles.size() == positionFiles.size() && paramFiles.size() == densitiesFiles.size());

	#pragma omp parallel for
	for (unsigned int t = 0; t < paramFiles.size(); t++) {

		vector<Real> params;
		vector<Vector3R> positions;
		vector<Real> densities;

		std::string filename = fileDir + sim_name + "_params_" + std::to_string(t) + ".cereal";
		load_scalars(filename, params);
		filename = fileDir + sim_name + "_positions_" + std::to_string(t) + ".cereal";
		load_vectors(filename, positions);
		filename = fileDir + sim_name + "_densities_" + std::to_string(t) + ".cereal";
		load_scalars(filename, densities);

		vector<size_t> fugitives;

		for (size_t particleID = 0; particleID < positions.size(); particleID ++) {

			bool inside = true;

			inside &= (lowerCorner(0) <= positions[particleID](0));
			inside &= (lowerCorner(1) <= positions[particleID](1));
			inside &= (lowerCorner(2) <= positions[particleID](2));

			inside &= (positions[particleID](0) <= upperCorner(0));
			inside &= (positions[particleID](1) <= upperCorner(1));
			inside &= (positions[particleID](2) <= upperCorner(2));

			if (!inside) fugitives.push_back(particleID);
		}
		std::reverse(fugitives.begin(), fugitives.end());

		for (auto particleID : fugitives) positions.erase(positions.begin() + particleID);
		for (auto particleID : fugitives) densities.erase(densities.begin() + particleID);

		auto fluid = new Fluid(params, positions, densities, initValue, lowerCorner, upperCorner, cubeResol);
		MarchingCubes mcb(lowerCorner, upperCorner, cubeResol);
		mcb.setObject(fluid);

		vector<Vector3R> triangle_mesh;

		mcb.getTriangleMesh(triangle_mesh);
		delete fluid;

		vector<array<int, 3>> triangles;

		for(int i = 0; i < triangle_mesh.size(); i += 3) triangles.push_back({i, i + 1, i + 2});

		std::string surface_filename = fileDir + sim_name + "_surface_" + std::to_string(t) + ".vtk";

		learnSPH::saveTriMeshToVTK(surface_filename, triangle_mesh, triangles);

		cout << "\nframe [" << t << "] rendered" << endl;
	}
}
