#pragma once
#include "ZhuBridsonReconstruction.hpp"
#include <unordered_map>
#include <types.hpp>

class SolenthilerReconstruction : public ZhuBridsonReconstruction
{
public:
	explicit SolenthilerReconstruction(std::shared_ptr<learnSPH::FluidSystem> fluid,
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
	explicit SolenthilerReconstruction(const SolenthilerReconstruction& other):
		ZhuBridsonReconstruction(other),
		mTLow(other.mTLow),
		mTHigh(other.mTHigh)
	{
	}
		
private:
	void updateLevelSet() override;
	void updateGrid() override;
	void configureHashTables() override;
	void updateFFunction();
	void updateGradientSums();
	void updateJakobians();

	float getSDFvalue(int i, int j, int k) const override;
	struct CellGradient {
		Matrix3d jakobian;
		Real fVal;
		Vector3R gradSum;
	};
	Real mTLow;
	Real mTHigh;
	std::unordered_map<int, CellGradient> mCellGradComponents;
	

};
