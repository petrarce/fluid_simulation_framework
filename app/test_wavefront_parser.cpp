#include <stdlib.h>     // rand
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // std::max

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <learnSPH/core/storage.h>
#include <types.hpp>
#include <learnSPH/core/particle_sampler.h>
#include <CompactNSearch>

#include <learnSPH/core/vtk_writer.h>
#include <learnSPH/simulation/solver.h>
#include <chrono>

using namespace learnSPH;

int main(int argc, char** argv)
{
	std::cout << "Welcome to the learnSPH framework!!" << std::endl;
	std::cout << "Generating a sample scene..." << std::endl;

	assert(argc == 14);

	string pathToWavefront = string(argv[1]);
	Vector3R rotationAxe = Vector3R(stod(argv[2]), stod(argv[3]), stod(argv[4]));
	Real angle = stod(argv[5])/360 * 2 * PI;
	Vector3R scale = Vector3R(stod(argv[6]), stod(argv[7]), stod(argv[8]));
	Vector3R translation = Vector3R(stod(argv[9]), stod(argv[10]), stod(argv[11]));
	Real samplingDist = stod(argv[12]);
	Real eta = stod(argv[13]);

	Matrix3d R;
	R = 	Eigen::AngleAxisd(angle, rotationAxe);
	Matrix3d S = Eigen::Matrix3d::Identity();
	S(0,0) = scale[0]; S(1,1) = scale[1]; S(2,2) = scale[2];
	Matrix4d transform;
	transform.block<3,3>(0,0) = S*R;
	transform.block<3,1>(0,3) = translation;
	transform(3,3) = 1;
	BorderSystem* borderModel = sample_border_model(transform, pathToWavefront, 1000, samplingDist, eta);

	std::string filename = "res/border_model.vtk";
	vector<Vector3R> dummyVector(borderModel->size());
	learnSPH::saveParticlesToVTK(filename, borderModel->getPositions(), borderModel->getVolumes(), dummyVector);

	delete borderModel;

	BorderSystem* borderConeParticles = sample_border_cone(1, Vector3R(0,0,0), 3, Vector3R(3,3,3), 1000, samplingDist, eta);

	filename = "res/border_cone.vtk";
	dummyVector.resize(borderConeParticles->size());
	learnSPH::saveParticlesToVTK(filename, borderConeParticles->getPositions(), borderConeParticles->getVolumes(), dummyVector);
	delete borderConeParticles;

	std::cout << "completed!" << std::endl;

	return 0;
}
