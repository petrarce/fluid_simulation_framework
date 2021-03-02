#pragma once
//std
#include <vector>
#include <mutex>
#include <unordered_map>

//learnSPH
#include <learnSPH/core/storage.h>

#include "NaiveMarchingCubes.hpp"

class MinDistReconstruction : public MarchingCubes
{
public:
	explicit MinDistReconstruction(std::shared_ptr<learnSPH::FluidSystem> fluid,
						   const Eigen::Vector3d lCorner,
						   const Eigen::Vector3d uCorner,
						   const Eigen::Vector3d cResolution,
						   float supportRadius):
		MarchingCubes(fluid, lCorner, uCorner, cResolution, supportRadius),
		mSupportRadius(supportRadius)
	{
	}
	
	explicit MinDistReconstruction(const MinDistReconstruction& other):
		MarchingCubes(other),
		mSupportRadius(other.mSupportRadius)
	{}

protected:

	void updateGrid() override
	{
		mSurfaceCells.clear();
		const auto& particles = mFluid->getPositions();
		for(int i = 0; i < mSurfaceParticlesCount; i++)
		{
			auto nCells = getNeighbourCells(particles[i], mSupportRadius, false);
			for(const auto& nc : nCells)
			{
				auto cI = cellIndex(nc);
				if(mSurfaceCells.find(cI) == mSurfaceCells.end())
				{
					mSurfaceCells[cI] = 1;
					mSurfaceCellsCurvature[cI] = mCurvature[i];
					mSDF[cI] = (cellCoord(nc) - particles[i]).norm();
				}
				else
				{
					mSurfaceCells[cI]++;
					mSurfaceCellsCurvature[cI] += mCurvature[i];
					Real newLength = (cellCoord(nc) - particles[i]).norm();
					if( newLength < mSDF[cI])
						mSDF[cI] = newLength;
				}
			}
		}
	}
	void updateLevelSet() override
	{
		//level set is already computed in the updateGrid()
	}
	void configureHashTables() override
	{
		MarchingCubes::configureHashTables();
		mSDF.max_load_factor(mSurfaceCells.max_load_factor());
		mSDF.rehash(mSurfaceCells.bucket_count());
	}
	bool getSDFvalue(int i, int j, int k, float& sdf) const override
	{
		auto cI = cellIndex(Eigen::Vector3i(i, j, k));
		auto sdfPtr = mSDF.find(cI);
		if(sdfPtr == mSDF.end())
			return false;
		sdf = sdfPtr->second - 0.5 * mSupportRadius;
		return true;
	}

	Real mSupportRadius;
	std::unordered_map<size_t, Real> mSDF;
};
