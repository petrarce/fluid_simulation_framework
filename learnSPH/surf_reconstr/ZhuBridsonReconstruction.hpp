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
		mRadii(radii)
	{
	}
	
	explicit ZhuBridsonReconstruction(const ZhuBridsonReconstruction& other ):
		MarchingCubes(other),
		denominators(other.denominators),
		dAvg(other.dAvg),
		xAvg(other.xAvg),
		mRadii(other.mRadii)
	{}

protected:
	std::unordered_map<int, Real> denominators;
	std::unordered_map<int, Real> dAvg;
	std::unordered_map<int, Vector3R> xAvg;
	Real mRadii;

	void updateGrid() override;
	void updateLevelSet() override;
	void configureHashTables() override;
	bool getSDFvalue(int i, int j, int k, float& sdf) const override;
	
	void updateDenominators();
	
	void updateAvgs();

};
