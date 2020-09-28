#pragma once
#include <types.hpp>
#include <learnSPH/core/vtk_writer.h>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <learnSPH/core/kernel.h>
#include <learnSPH/surf_reconstr/ZhuBridsonReconstruction.hpp>
#include <learnSPH/surf_reconstr/NaiveMarchingCubes.hpp>
#include <learnSPH/surf_reconstr/SolenthilerReconstruction.hpp>


template<class BaseClass, class... Args>
class BlurredReconstruction : public BaseClass
{
public:
	explicit BlurredReconstruction(
			Args... args,
			float smoothingFactor,
			int kernelSize,
			int offset):
		BaseClass(args...),
		mSmoothingFactor(smoothingFactor),
		mKernelSize(kernelSize),
		mOffset(offset)
	{
	}
	
	explicit BlurredReconstruction(const BlurredReconstruction& other):
		BaseClass(other),
		mSmoothingFactor(other.mSmoothingFactor),
		mKernelSize(other.mKernelSize),
		mOffset(other.mKernelSize)
	{}
private:
	void configureHashTables() override
	{
		BaseClass::configureHashTables();
		mLevelSetValues.max_load_factor(BaseClass::mSurfaceCells.max_load_factor());
	}
	
	void updateGrid() override
	{
		BaseClass::updateGrid();
		mLevelSetValues.clear();
	}
	
	void updateLevelSet() override
	{
		BaseClass::updateLevelSet();
		saveLevelSetValues();
#ifdef DEBUG
		static int cnt = 0;
		std::vector<Vector3R> points;
		std::vector<Real> sdf;
		for(const auto& item : BaseClass::mSurfaceCells)
		{
			points.push_back(BaseClass::cellCoord(BaseClass::cell(item.first)));
			sdf.push_back(mLevelSetValues[item.first]);
		}
		learnSPH::saveParticlesToVTK("/tmp/SdfBeforeBlur" + to_string(cnt) + ".vtk", points, sdf);
#endif
		blurLevelSet(mKernelSize, mOffset);
#ifdef DEBUG
		sdf.clear();
		for(const auto& item : BaseClass::mSurfaceCells)
			sdf.push_back(mLevelSetValues[item.first]);
		learnSPH::saveParticlesToVTK("/tmp/SdfAfterBlur" + to_string(cnt) + ".vtk", points, sdf);
		cnt++;
#endif
	}
	float getSDFvalue(int i, int j, int k) const override
	{
		auto cI = BaseClass::cellIndex(Eigen::Vector3i(i,j,k));
		auto sdfItem = mLevelSetValues.find(cI);
		if(sdfItem == mLevelSetValues.end())
			throw std::invalid_argument("cell is not in the domain");
		
		return sdfItem->second;
	}
	
	void saveLevelSetValues()
	{
		for(const auto& item : BaseClass::mSurfaceCells)
		{
			auto c = BaseClass::cell(item.first);
			mLevelSetValues[item.first] = BaseClass::getSDFvalue(c(0), c(1), c(2));
		}
	}
	
	void blurLevelSet(int kernelSize, int offset)
	{
		auto newLevelSet = mLevelSetValues;
		Real maxRadii = BaseClass::mResolution(0) * offset * kernelSize * 1.1;
		for(const auto& cellItem : BaseClass::mSurfaceCells)
		{
			auto cI = cellItem.first;
			auto c =BaseClass:: cell(cI);
			auto cC = BaseClass::cellCoord(c);
			Real dfValue = 0;
			auto nbs = getNeighbourCells(c, kernelSize, offset);
			Real wSum = 0;
			for(const auto& nb : nbs)
			{
				float sdfVal = 0;
				try{ sdfVal = getSDFvalue(nb(0), nb(1), nb(2)); }
				catch(...){ sdfVal = getSDFvalue(c(0), c(1), c(2)); }
				Real w = learnSPH::kernel::kernelFunction(BaseClass::cellCoord(nb), cC, maxRadii);
				wSum += w;
				dfValue += sdfVal * w;
			}
			dfValue /= wSum;
			newLevelSet[cI] = mLevelSetValues[cI] * (1 - mSmoothingFactor) + mSmoothingFactor * dfValue;
		}
		mLevelSetValues.swap(newLevelSet);
	}
	
	std::vector<Eigen::Vector3i> getNeighbourCells(Eigen::Vector3i baseCell, int kernelSize, int offset)
	{
		std::vector<Eigen::Vector3i> neighbors;
		
		for(int i = -kernelSize * offset; i <= kernelSize * offset; i += offset)
			for(int j = -kernelSize * offset; j <= kernelSize * offset; j += offset)
				for(int k = -kernelSize * offset; k <= kernelSize * offset; k += offset)
					neighbors.push_back(baseCell + Eigen::Vector3i(i,j,k));
		
		return neighbors;
	}
	
	float	mSmoothingFactor	{ 1 };
	size_t	mKernelSize			{ 1 };
	size_t	mOffset				{ 1 };
	std::unordered_map<size_t, Real> mLevelSetValues;
};

typedef  BlurredReconstruction<ZhuBridsonReconstruction, std::shared_ptr<learnSPH::FluidSystem> , const Eigen::Vector3d , 
								const Eigen::Vector3d , const Eigen::Vector3d , float > ZhuBridsonBlurred;
typedef  BlurredReconstruction<SolenthilerReconstruction, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, const Eigen::Vector3d, 
								const Eigen::Vector3d , float, float, float> SolenthilerBlurred;
typedef  BlurredReconstruction<NaiveMarchingCubes, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, 
								const Eigen::Vector3d, const Eigen::Vector3d, float> NaiveBlurred;


