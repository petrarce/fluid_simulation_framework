#pragma once
//std
#include <vector>
#include <mutex>

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
	std::vector<Real> denominators;
	std::vector<Real> rAvg;
	std::vector<Vector3R> xAvg;
	Real mRadii;

	void updateGrid() override
	{
			denominators.assign(denominators.size(), 0.f);
			rAvg.assign(rAvg.size(), 0.f);
			for(size_t i = 0; i < xAvg.size(); i++)
				xAvg[i] = Vector3R(0,0,0);
	}
	void updateLevelSet() override
	{
		updateDenominators();
		updateAvgs();
	}
	float getSDFvalue(int i, int j, int k) const override
	{
		auto cell = Eigen::Vector3i(i,j,k);
		return -1 * ((cellCoord(cell) - xAvg[cellIndex(cell)]).norm() - rAvg[cellIndex(cell)]);
	}
	
	void updateDenominators()
	{
		const vector<Vector3R>& particles = mFluid->getPositions();
	
		for(size_t i = 0; i < particles.size(); i++)
		{
			std::vector<Vector3i> neighbours = getNeighbourCells(particles[i], mRadii);
			for(const auto& cell : neighbours)
				denominators[cellIndex(cell)] += learnSPH::kernel::kernelCubic(cellCoord(cell), particles[i], mRadii);
		}
	}
	
	void updateAvgs()
	{
		const std::vector<Vector3R>& particles = mFluid->getPositions();
	
		for(size_t i = 0; i < particles.size(); i++)
		{
			std::vector<Vector3i> neighbours = getNeighbourCells(particles[i], mRadii);
			for(const auto& cell : neighbours)
			{
				auto weight = learnSPH::kernel::kernelCubic(cellCoord(cell), particles[i], mRadii);
				xAvg[cellIndex(cell)] += weight / denominators[cellIndex(cell)] * particles[i];
				rAvg[cellIndex(cell)] += weight / denominators[cellIndex(cell)] * mFluid->getSmoothingLength();
			}
		}
	}

};
