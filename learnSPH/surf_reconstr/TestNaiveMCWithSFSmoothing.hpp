#pragma once
#include "ZhuBridsonReconstruction.hpp"
#include <learnSPH/core/vtk_writer.h>
#define DEBUG

class TestNaiveMCWithSFSmoothing : public ZhuBridsonReconstruction
{
public:
	explicit TestNaiveMCWithSFSmoothing(
			std::shared_ptr<learnSPH::FluidSystem> fluid,
			const Eigen::Vector3d lCorner,
			const Eigen::Vector3d uCorner,
			const Eigen::Vector3d cResolution,
			float radii,
			float smoothingFactor,
			int kernelSize,
			int offset):
		ZhuBridsonReconstruction(fluid, lCorner, uCorner, cResolution, radii),
		mSmoothingFactor(smoothingFactor),
		mKernelSize(kernelSize),
		mOffset(offset)
	{
	}
	
	explicit TestNaiveMCWithSFSmoothing(const TestNaiveMCWithSFSmoothing& other):
		ZhuBridsonReconstruction(other),
		mSmoothingFactor(other.mSmoothingFactor),
		mKernelSize(other.mKernelSize),
		mOffset(other.mKernelSize)
	{}
private:
	void configureHashTables() override
	{
		ZhuBridsonReconstruction::configureHashTables();
		mLevelSetValues.max_load_factor(mSurfaceCells.max_load_factor());
	}
	
	void updateGrid() override
	{
		ZhuBridsonReconstruction::updateGrid();
		mLevelSetValues.clear();
	}
	
	void updateLevelSet() override
	{
		ZhuBridsonReconstruction::updateLevelSet();
		saveLevelSetValues();
#ifdef DEBUG
		static int cnt = 0;
		std::vector<Vector3R> points;
		std::vector<Real> sdf;
		for(const auto& item : mSurfaceCells)
		{
			points.push_back(cellCoord(cell(item.first)));
			sdf.push_back(mLevelSetValues[item.first]);
		}
		learnSPH::saveParticlesToVTK("/tmp/SdfBeforeBlur" + to_string(cnt) + ".vtk", points, sdf);
#endif
		blurLevelSet(mKernelSize, mOffset);
#ifdef DEBUG
		sdf.clear();
		for(const auto& item : mSurfaceCells)
			sdf.push_back(mLevelSetValues[item.first]);
		learnSPH::saveParticlesToVTK("/tmp/SdfAfterBlur" + to_string(cnt) + ".vtk", points, sdf);
		cnt++;
#endif
	}
	float getSDFvalue(int i, int j, int k) const override
	{
		auto cI = cellIndex(Vector3i(i,j,k));
		auto sdfItem = mLevelSetValues.find(cI);
		if(sdfItem == mLevelSetValues.end())
			throw std::invalid_argument("cell is not in the domain");
		
		return sdfItem->second;
	}
	
	void saveLevelSetValues()
	{
		for(const auto& item : mSurfaceCells)
		{
			auto c = cell(item.first);
			mLevelSetValues[item.first] = ZhuBridsonReconstruction::getSDFvalue(c(0), c(1), c(2));
		}
	}
	
	void blurLevelSet(int kernelSize, int offset)
	{
		auto newLevelSet = mLevelSetValues;
		Real maxRadii = mResolution(0) * offset * kernelSize * 1.1;
		for(const auto& cellItem : mSurfaceCells)
		{
			auto cI = cellItem.first;
			auto c = cell(cI);
			auto cC = cellCoord(c);
			Real dfValue = 0;
			auto nbs = getNeighbourCells(cellCoord(c), kernelSize, offset);
			Real wSum = 0;
			for(const auto& nb : nbs)
			{
				float sdfVal = 0;
				try{ sdfVal = getSDFvalue(nb(0), nb(1), nb(2)); }
				catch(...){ sdfVal = getSDFvalue(c(0), c(1), c(2)); }
				Real w = learnSPH::kernel::kernelFunction(cellCoord(nb), cC, maxRadii);
				wSum += w;
				dfValue += sdfVal * w;
			}
			dfValue /= wSum;
			newLevelSet[cI] = mLevelSetValues[cI] * (1 - mSmoothingFactor) + mSmoothingFactor * dfValue;
		}
		mLevelSetValues.swap(newLevelSet);
	}
	
	std::vector<Vector3i> getNeighbourCells(Eigen::Vector3d position, int kernelSize, int offset)
	{
		std::vector<Eigen::Vector3i> neighbors;
		auto baseCell = cell(position);
		for(int i = -kernelSize * offset; i <= kernelSize * offset; i += offset)
			for(int j = -kernelSize * offset; j <= kernelSize * offset; j += offset)
				for(int k = -kernelSize * offset; k <= kernelSize * offset; k += offset)
					neighbors.push_back(baseCell + Eigen::Vector3i(i,j,k));
		
		return neighbors;
	}
	
	float	mSmoothingFactor	{ 1 };
	size_t	mKernelSize			{ 1 };
	size_t	mOffset				{ 1 };
	unordered_map<size_t, Real> mLevelSetValues;
};
