#pragma once
//std
#include <vector>
#include <mutex>
#include <unordered_map>

//learnSPH
#include <learnSPH/core/storage.h>

#include "NaiveMarchingCubes.hpp"

class ZhuBridsonReconstruction : public MarchingCubes
{
public:
	explicit ZhuBridsonReconstruction(std::shared_ptr<learnSPH::FluidSystem> fluid,
						   const Eigen::Vector3d lCorner,
						   const Eigen::Vector3d uCorner,
						   const Eigen::Vector3d cResolution,
						   float radii):
		MarchingCubes(fluid, lCorner, uCorner, cResolution, radii),
		denominators(mDimentions(0) * mDimentions(1) * mDimentions(2)),
		rAvg(mDimentions(0) * mDimentions(1) * mDimentions(2)),
		xAvg(mDimentions(0) * mDimentions(1) * mDimentions(2)),
		mRadii(radii)
	{
	}
	
	explicit ZhuBridsonReconstruction(const ZhuBridsonReconstruction& other ):
		MarchingCubes(other),
		denominators(other.denominators),
		rAvg(other.rAvg),
		xAvg(other.xAvg),
		mRadii(other.mRadii)
	{}

protected:
	std::unordered_map<int, Real> denominators;
	std::unordered_map<int, Real> rAvg;
	std::unordered_map<int, Vector3R> xAvg;
	Real mRadii;

	void updateGrid() override;
	void updateLevelSet() override;
	void configureHashTables() override;
	float getSDFvalue(int i, int j, int k) const override;
	
	void updateDenominators();
	
	void updateAvgs();

};
