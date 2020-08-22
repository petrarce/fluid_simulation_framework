#pragma once
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <learnSPH/core/storage.h>

class SurfaceReconstructor
{
	
public:
    virtual std::vector<Eigen::Vector3d> generateMesh(const std::shared_ptr<learnSPH::FluidSystem> fluid) = 0;
};
