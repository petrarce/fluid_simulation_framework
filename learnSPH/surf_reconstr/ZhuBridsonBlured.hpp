#pragma once
#include <types.hpp>
#include <learnSPH/core/vtk_writer.h>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <random>
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
			int offset,
			float depth):
		BaseClass(args...),
		mSmoothingFactor(smoothingFactor),
		mKernelSize(kernelSize),
		mOffset(offset),
		mKernelDepth(depth)
	{
	}
	
	explicit BlurredReconstruction(const BlurredReconstruction& other):
		BaseClass(other),
		mSmoothingFactor(other.mSmoothingFactor),
		mKernelSize(other.mKernelSize),
		mOffset(other.mKernelSize),
		mKernelDepth(other.mKernelDepth)
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
		std::vector<Vector3R> sdfGradients;
		for(const auto& item : BaseClass::mSurfaceCells)
		{
			points.push_back(BaseClass::cellCoord(BaseClass::cell(item.first)));
			sdf.push_back(mLevelSetValues[item.first]);
			sdfGradients.push_back(getSDFGrad(BaseClass::cell(item.first)));
		}
		learnSPH::saveParticlesToVTK("/tmp/SdfBeforeBlur" + to_string(cnt) + ".vtk", points, sdf, sdfGradients);
#endif
		blurLevelSet(mKernelSize, mOffset, mKernelDepth);
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
	
	void blurLevelSet(int kernelSize, int offset, Real depth)
	{
		auto newLevelSet = mLevelSetValues;
		Real maxRadii = BaseClass::mResolution(0) * offset * kernelSize * 1.1;
		for(const auto& cellItem : BaseClass::mSurfaceCells)
		{
			auto cI = cellItem.first;
			auto c =BaseClass:: cell(cI);
			auto cC = BaseClass::cellCoord(c);
			Real dfValue = 0;
			Real cellSdf = MarchingCubes::getSDFvalue(c);
			auto nbs = getNeighbourCells(c, kernelSize, offset, depth);
			Real wSum = 0;
			for(const auto& nb : nbs)
			{
				float sdfVal = 0;
				try {sdfVal = getSDFvalue(nb(0), nb(1), nb(2)); }
				catch(...) {sdfVal = cellSdf;}
				Real w = learnSPH::kernel::kernelFunction(BaseClass::cellCoord(nb), cC, maxRadii);
				wSum += w;
				dfValue += sdfVal * w;
			}
			dfValue /= wSum;
			Real smoothFactor = std::min(1.f, mSmoothingFactor * static_cast<float>(cellItem.second) / BaseClass::mPartPerSupportArea);
			newLevelSet[cI] = mLevelSetValues[cI] * (1 - smoothFactor) + smoothFactor * dfValue;
		}
		mLevelSetValues.swap(newLevelSet);
	}
	
	Eigen::Vector3d getSDFGrad(const Vector3i& c) const
	{
		float cellSDFval = MarchingCubes::getSDFvalue(c);
		float sdfValX = 0.f;
		float sdfValY = 0.f;
		float sdfValZ = 0.f;
		try {sdfValX = MarchingCubes::getSDFvalue(c - Vector3i(1, 0, 0));}
		catch(...){sdfValX = cellSDFval;}
		try {sdfValY = MarchingCubes::getSDFvalue(c - Vector3i(0, 1, 0));}
		catch(...){sdfValY = cellSDFval;}
		try {sdfValZ = MarchingCubes::getSDFvalue(c - Vector3i(0, 0, 1));}
		catch(...){sdfValZ = cellSDFval;}
		float dx = (cellSDFval - sdfValX) / BaseClass::mResolution(0);
		float dy = (cellSDFval - sdfValY) / BaseClass::mResolution(1);
		float dz = (cellSDFval - sdfValZ) / BaseClass::mResolution(2);
		
		return Eigen::Vector3d(dx, dy, dz);
	}
	
	std::vector<Eigen::Vector3i> getNeighbourCells(Eigen::Vector3i baseCell, int kernelSize, int offset, Real depth)
	{
#ifdef DEBUG
		static std::mt19937 generator(1);
		static std::uniform_real_distribution<float> distr {0, 1};
		static auto dice = std::bind(distr, generator);
		std::vector<Vector3R> cells;
		float probability = std::min(1., 1000. / BaseClass::mSurfaceCells.size());
		static size_t cnt = 0;
#endif
		std::vector<Eigen::Vector3i> neighbors;
		Eigen::Vector3d grad = getSDFGrad(baseCell);
		if(grad.dot(grad) < 1e-6)
			return neighbors;
		grad.normalize();
		
		for(int i = -kernelSize * offset; i <= kernelSize * offset; i += offset)
			for(int j = -kernelSize * offset; j <= kernelSize * offset; j += offset)
				for(int k = -kernelSize * offset; k <= kernelSize * offset; k += offset)
				{
					if(Eigen::Vector3i(i,j,k) == Eigen::Vector3i(0,0,0))
						neighbors.push_back(baseCell);
					
					//if projection of the point offset to gradient in the point is larger, that half of the kernel - dont take the point for bluering while 
					Eigen::Vector3d offsetVector = BaseClass::cellCoord(baseCell) - BaseClass::cellCoord(baseCell + Eigen::Vector3i(i,j,k));
					if(std::fabs(offsetVector.dot(grad)) > (depth * (kernelSize) * BaseClass::mResolution(0)))
						continue;
					
					neighbors.push_back(baseCell + Eigen::Vector3i(i,j,k));
#ifdef DEBUG
					cells.push_back(BaseClass::cellCoord(baseCell + Eigen::Vector3i(i,j,k)));
#endif
				}
		
#ifdef DEBUG
		if(dice() < probability)
		{
			cells.push_back(BaseClass::cellCoord(baseCell));
			std::vector<Real> dencities(cells.size() - 1, 0);
			dencities.push_back(1);
			learnSPH::saveParticlesToVTK("/tmp/GradNeighbours" + to_string(cnt) + ".vtk", cells, dencities, std::vector<Vector3R>(cells.size(), grad));
			cnt++;
		}
#endif
		return neighbors;
	}
	
	float	mSmoothingFactor	{ 1 };
	size_t	mKernelSize			{ 1 };
	size_t	mOffset				{ 1 };
	float	mKernelDepth		{ 0.5 };
	std::unordered_map<size_t, Real> mLevelSetValues;
};

typedef  BlurredReconstruction<ZhuBridsonReconstruction, std::shared_ptr<learnSPH::FluidSystem> , const Eigen::Vector3d , 
								const Eigen::Vector3d , const Eigen::Vector3d , float > ZhuBridsonBlurred;
typedef  BlurredReconstruction<SolenthilerReconstruction, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, const Eigen::Vector3d, 
								const Eigen::Vector3d , float, float, float> SolenthilerBlurred;
typedef  BlurredReconstruction<NaiveMarchingCubes, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, 
								const Eigen::Vector3d, const Eigen::Vector3d, float> NaiveBlurred;


