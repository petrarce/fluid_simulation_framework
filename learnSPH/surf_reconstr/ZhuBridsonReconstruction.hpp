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

	void updateGrid() override;
	void updateLevelSet() override;
	void clearBuffers() override
	{
		denominators.clear(); denominators.shrink_to_fit();
		dAvg.clear(); dAvg.shrink_to_fit();
		xAvg.clear(); xAvg.shrink_to_fit();
	}

	Real mRadii;
	std::vector<Real> denominators;
	std::vector<Real> dAvg;
	std::vector<Vector3R> xAvg;

protected:
	void updateDenominators();
	void updateAvgs();

};
