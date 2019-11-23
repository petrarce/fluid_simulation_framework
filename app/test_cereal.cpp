#include <string>
#include <fstream>
#include <iostream>
#include <types.hpp>
#include <cassert>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

void save_vectors(const std::string &path, std::vector<Vector3R> &data)
{
	std::ofstream os(path, std::ios::binary);

	cereal::BinaryOutputArchive boa(os);

	boa(data.size());

	for (Vector3R &vec : data) boa(vec(0), vec(1), vec(2));
}

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

void save_scalars(const std::string &path, std::vector<Real> &data)
{
	std::ofstream os(path, std::ios::binary);

	cereal::BinaryOutputArchive boa(os);

	boa(data.size());

	for (auto val : data) boa(val);
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

int main() {

	std::vector<Real> scalars;
	std::vector<Vector3R> vectors;

	scalars.push_back(0.1);
	scalars.push_back(2.3);
	scalars.push_back(4.5);

	vectors.push_back(Vector3R(0.0, 1.0, 2.0));
	vectors.push_back(Vector3R(0.5, 1.5, 2.5));

	save_scalars(std::string("./scalars.cereal"), scalars);
	save_vectors(std::string("./vectors.cereal"), vectors);

	std::vector<Real> reload_scalars;
	std::vector<Vector3R> reload_vectors;

	load_scalars(std::string("./scalars.cereal"), reload_scalars);
	load_vectors(std::string("./vectors.cereal"), reload_vectors);

	assert(scalars[1] == reload_scalars[1]);

	assert((vectors[1] - reload_vectors[1]).norm() <= 1e-6);

	return 0;
}