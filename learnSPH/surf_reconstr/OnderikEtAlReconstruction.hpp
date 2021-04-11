#pragma once
#include "NaiveMarchingCubes.hpp"
#include "ZhuBridsonReconstruction.hpp"
#include <vector>
#include <Eigen/Dense>

class OnderikEtAlReconstruction : public ZhuBridsonReconstruction
{
	float mWmin {0};
	float mWmax {1.05};
	std::vector<float> mWeightedSums;
	std::vector<Vector3R> mWeightedAvgs;
public:
	explicit OnderikEtAlReconstruction(std::shared_ptr<learnSPH::FluidSystem> fluid,
						   const Eigen::Vector3d lCorner,
						   const Eigen::Vector3d uCorner,
						   const Eigen::Vector3d cResolution,
						   float radii,
						   float wMin,
						   float wMax):
		ZhuBridsonReconstruction(fluid, lCorner, uCorner, cResolution, radii),
		mWmin(wMin),
		mWmax(wMax)
	{
	}
	explicit OnderikEtAlReconstruction(const OnderikEtAlReconstruction& other):
		ZhuBridsonReconstruction(other),
		mWmin(other.mWmin),
		mWmax(other.mWmax),
		mWeightedSums(other.mWeightedSums),
		mWeightedAvgs(other.mWeightedAvgs)
	{}
protected:
	void updateLevelSet() override;
private:
	void updateWeightedSums();
	float g(float w);
	void updateSdf();
};
