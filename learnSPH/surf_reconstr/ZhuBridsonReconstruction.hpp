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
	{}
	
	explicit ZhuBridsonReconstruction(const ZhuBridsonReconstruction& other ):
		MarchingCubes(other),
		denominators(other.denominators),
		rAvg(other.rAvg),
		xAvg(other.xAvg),
		mRadii(other.mRadii)
	{}

private:
	std::unordered_map<int, Real> denominators;
	std::unordered_map<int, Real> rAvg;
	std::unordered_map<int, Vector3R> xAvg;
	Real mRadii;

	void updateGrid() override
	{
		rAvg.clear();
		xAvg.clear();
		denominators.clear();
		mSurfaceCells.clear();
		const auto& particles = mFluid->getPositions();
		for(int i = 0; i < mSurfaceParticlesCount; i++)
		{
			auto nCells = getNeighbourCells(particles[i], mRadii, false);
			for(const auto& nc : nCells)
				mSurfaceCells[cellIndex(nc)] = cellIndex(nc);
		}
	}
	void updateLevelSet() override
	{
		updateDenominators();
		updateAvgs();
	}
	float getSDFvalue(int i, int j, int k) const override
	{
		auto cell = Eigen::Vector3i(i,j,k);
		auto cI = cellIndex(cell);
		auto cC = cellCoord(cell);
		auto xAvgI = xAvg.find(cI);
		auto rAvgI = rAvg.find(cI);
		if(xAvgI == xAvg.end())
			//TODO compute some acceptable value
			return -1;
		return -1 * ((cC - xAvgI->second).norm() - rAvgI->second);
	}
	
	void updateDenominators()
	{
		const vector<Vector3R>& particles = mFluid->getPositions();
	
		for(size_t i = 0; i < particles.size(); i++)
		{
			std::vector<Vector3i> neighbours = getNeighbourCells(particles[i], mRadii);
			for(const auto& cell : neighbours){
				auto cI = cellIndex(cell);
				auto cC = cellCoord(cell);
				auto denominatorsI = denominators.find(cI);
				if(denominatorsI == denominators.end())
					denominators[cI] = 0;
				denominators[cI] += learnSPH::kernel::kernelCubic(cC, particles[i], mRadii);
			}
		}
	}
	
	void updateAvgs()
	{
		const std::vector<Vector3R>& particles = mFluid->getPositions();
	
		for(size_t i = 0; i < particles.size(); i++)
		{
			std::vector<Vector3i> neighbours = getNeighbourCells(particles[i], mRadii);
			Real minDist = std::numeric_limits<Real>::max();
			for(const auto& cell : neighbours)
			{
				auto cI = cellIndex(cell);
				auto cC = cellCoord(cell);
				auto xAvgI = xAvg.find(cI);
				auto rAvgI = rAvg.find(cI);
				if(xAvg.find(cI) == xAvg.end())
				{
					xAvg[cI] = Vector3R(0,0,0);
					rAvg[cI] = 0;
				}
				auto weight = learnSPH::kernel::kernelCubic(cC, particles[i], mRadii);
				xAvg[cI] += weight / denominators[cI] * particles[i];
				rAvg[cI] += weight / denominators[cI] * mFluid->getSmoothingLength();
			}
		}
	}

};
