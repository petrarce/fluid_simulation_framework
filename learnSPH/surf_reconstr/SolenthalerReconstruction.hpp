#pragma once
#include "ZhuBridsonReconstruction.hpp"
#include <unordered_map>
#include <types.hpp>

class SolenthalerReconstruction : public ZhuBridsonReconstruction
{
public:
	explicit SolenthalerReconstruction(std::shared_ptr<learnSPH::FluidSystem> fluid,
							  const Eigen::Vector3d lCorner,
							  const Eigen::Vector3d uCorner,
							  const Eigen::Vector3d cResolution,
							  float radii,
							  float tLow,
							  float tHigh):
		ZhuBridsonReconstruction(fluid, lCorner, uCorner, cResolution, radii),
		mTLow(tLow),
		mTHigh(tHigh)
	{}
	explicit SolenthalerReconstruction(const SolenthalerReconstruction& other):
		ZhuBridsonReconstruction(other),
		mTLow(other.mTLow),
		mTHigh(other.mTHigh)
	{
	}
		
private:
	void updateLevelSet() override;
	void updateGrid() override;
	void updateFFunction();
	void updateGradientSums();
	void updateJakobians();

	Real mTLow;
	Real mTHigh;
	std::vector<Matrix3d> mJacobians;
	std::vector<Real> mFVal;
	std::vector<Vector3R> mGradSums;
	

};
