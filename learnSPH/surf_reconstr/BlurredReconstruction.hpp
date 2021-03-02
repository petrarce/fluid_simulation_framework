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
#include <cassert>

#if 0
#ifndef DBG
#define DBG
#endif
#endif
template<class BaseClass, class... Args>
class BlurredReconstruction : public BaseClass
{
public:
	explicit BlurredReconstruction(
			Args... args,
			float smoothingFactor,
			int kernelSize,
			int offset,
			float depth,
			bool blurSurfaceCellsOnly,
			size_t blurIterations):
		BaseClass(args...),
		mSmoothingFactor(smoothingFactor),
		mKernelSize(kernelSize),
		mOffset(offset),
		mKernelDepth(depth),
		mBlurrSurfaceCellsOnly(blurSurfaceCellsOnly),
		mBlurIterations(blurIterations)
	{
	}
	
	explicit BlurredReconstruction(const BlurredReconstruction& other):
		BaseClass(other),
		mSmoothingFactor(other.mSmoothingFactor),
		mKernelSize(other.mKernelSize),
		mOffset(other.mKernelSize),
		mKernelDepth(other.mKernelDepth),
		mBlurrSurfaceCellsOnly(other.mBlurrSurfaceCellsOnly),
		mBlurIterations(other.mBlurIterations)
	{}
private:
	void configureHashTables() override
	{
		BaseClass::configureHashTables();
		mLevelSetValues.max_load_factor(BaseClass::mSurfaceCells.max_load_factor());
		mLevelSetValues.rehash(BaseClass::mSurfaceCells.bucket_count());
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
#ifdef DBG
		std::vector<Vector3R> points;
		std::vector<Real> sdf;
		std::vector<Vector3R> sdfGradients;
		for(const auto& item : BaseClass::mSurfaceCells)
		{
			points.push_back(BaseClass::cellCoord(BaseClass::cell(item.first)));
			sdf.push_back(mLevelSetValues[item.first]);
			sdfGradients.push_back(getSDFGrad(BaseClass::cell(item.first)));
		}
		learnSPH::saveParticlesToVTK("/tmp/SdfBeforeBlur" + BaseClass::mFrameNumber + ".vtk", points, sdf, sdfGradients);
#endif
		for(int i = 0; i < mBlurIterations; i++)
			blurLevelSet(mKernelSize, mOffset, mKernelDepth);
#ifdef DBG
		sdf.clear();
		for(const auto& item : BaseClass::mSurfaceCells)
			sdf.push_back(mLevelSetValues[item.first]);
		learnSPH::saveParticlesToVTK("/tmp/SdfAfterBlur" + BaseClass::mFrameNumber + ".vtk", points, sdf);
#endif
	}
	bool getSDFvalue(int i, int j, int k, float& sdf) const override
	{
		auto cI = BaseClass::cellIndex(Eigen::Vector3i(i,j,k));
		auto sdfItem = mLevelSetValues.find(cI);
		if(sdfItem == mLevelSetValues.end())
			return false;
		
		sdf = sdfItem->second;
		return true;
	}
	
	void saveLevelSetValues()
	{
		for(const auto& item : BaseClass::mSurfaceCells)
		{
			auto c = BaseClass::cell(item.first);
			float sdfVal;
			bool res = BaseClass::getSDFvalue(c(0), c(1), c(2), sdfVal);
			assert(res);
			mLevelSetValues[item.first] = sdfVal;
		}
	}
	
	void blurLevelSet(int kernelSize, int offset, Real depth)
	{
		auto newLevelSet = mLevelSetValues;
		Real maxRadii = BaseClass::mResolution(0) * offset * kernelSize * 1.1;
		typedef decltype(BaseClass::mSurfaceCells) CellsContainer;
		const CellsContainer* cellVertices = &this->mSurfaceCells;
#ifdef DBG
		static std::mt19937 generator(1);
		static std::uniform_real_distribution<float> distr {0, 1};
		static auto dice = std::bind(distr, generator);
		float probability = std::min(1., 1000. / BaseClass::mSurfaceCells.size());
		int cellsProcessed = 0;
		std::vector<Real> smoothFactors;
		std::vector<Real> curvatures;
		std::vector<Vector3R> points;
#endif
		if(mBlurrSurfaceCellsOnly)
			cellVertices = new CellsContainer(BaseClass::computeIntersectionCellVertices(mKernelSize));
		for(const auto& cellItem : *cellVertices)
		{
#ifdef DBG
			std::vector<Vector3R> nbCoords;
#endif
			auto cI = cellItem.first;
			auto c =BaseClass::cell(cI);
			auto cC = BaseClass::cellCoord(c);
			Real dfValue = 0;
			float cellSdf; bool res = MarchingCubes::getSDFvalue(c, cellSdf);
			assert(res);
			auto nbs = getNeighbourCells(c, kernelSize, offset, depth);
			Real wSum = 0;
			for(const auto& nb : nbs)
			{
				float sdfVal = 0;
				if(!getSDFvalue(nb(0), nb(1), nb(2), sdfVal))
					sdfVal = cellSdf;
				wSum += 1.;
				dfValue += sdfVal;
#ifdef DBG
				nbCoords.push_back(BaseClass::cellCoord(nb));
#endif

			}

			dfValue /= (wSum + 1e-6);
			Real smoothFactor = std::min(1.f, mSmoothingFactor * static_cast<float>(cellItem.second) / BaseClass::mPartPerSupportArea);
			smoothFactor = -1 * std::pow(1 - smoothFactor*smoothFactor, 10.) + 1;
			newLevelSet[cI] = mLevelSetValues[cI] * (1 - smoothFactor) + smoothFactor * dfValue;
#ifdef DBG
			if(dice() < probability)
			{
				nbCoords.push_back(cC);
				std::vector<Real> dencities(nbCoords.size() - 1, 0);
				dencities.push_back(1);
				learnSPH::saveParticlesToVTK("/tmp/GradNeighbours" + string("_") + BaseClass::mFrameNumber + "_" + to_string(cellsProcessed) + ".vtk",
											 nbCoords,
											 std::vector<Vector3R>(nbCoords.size(), getSDFGrad(c)));
				cellsProcessed++;
			}
			smoothFactors.push_back(smoothFactor);
			points.push_back(cC);
			curvatures.push_back(getCurvature(c));
#endif
		}
		mLevelSetValues.swap(newLevelSet);
		if(mBlurrSurfaceCellsOnly)
			delete cellVertices;
#ifdef DBG
		learnSPH::saveParticlesToVTK("/tmp/ParticleCountSmoothFactor_" + BaseClass::mFrameNumber + ".vtk", points, smoothFactors);
		learnSPH::saveParticlesToVTK("/tmp/Curvature" + BaseClass::mFrameNumber + ".vtk", points, curvatures);
#endif
	}
	
	Eigen::Vector3d getSDFGrad(const Vector3i& c) const
	{
		float cellSDFval;
		bool res = MarchingCubes::getSDFvalue(c, cellSDFval);
		assert(res);
		float sdfValX = 0.f;
		float sdfValY = 0.f;
		float sdfValZ = 0.f;
		if(!MarchingCubes::getSDFvalue(c - Vector3i(1, 0, 0), sdfValX))
			sdfValX = cellSDFval;
		if(!MarchingCubes::getSDFvalue(c - Vector3i(0, 1, 0), sdfValY))
			sdfValY = cellSDFval;
		if(!MarchingCubes::getSDFvalue(c - Vector3i(0, 0, 1), sdfValZ))
			sdfValZ = cellSDFval;
		float dx = (cellSDFval - sdfValX) / BaseClass::mResolution(0);
		float dy = (cellSDFval - sdfValY) / BaseClass::mResolution(1);
		float dz = (cellSDFval - sdfValZ) / BaseClass::mResolution(2);
		
		return Eigen::Vector3d(dx, dy, dz);
	}

	inline bool getSDFderivX(const Vector3i& c, float& deriv) const
	{
		float sdf1;
		if(!getSDFvalue(c(0), c(1), c(2), sdf1))
			return false;
		float sdf2;
		if(!getSDFvalue(c(0) - 1, c(1), c(2), sdf2))
			sdf2 = sdf1;

		deriv = sdf1 - sdf2 / BaseClass::mResolution(0);
		return true;
	}

	inline bool getSDFderivY(const Vector3i& c, float& deriv) const
	{
		float sdf1;
		if(!getSDFvalue(c(0), c(1), c(2), sdf1))
			return false;
		float sdf2;
		if(!getSDFvalue(c(0), c(1) - 1, c(2), sdf2))
			sdf2 = sdf1;

		deriv = sdf1 - sdf2 / BaseClass::mResolution(1);
		return true;
	}

	inline bool getSDFderivZ(const Vector3i& c, float& deriv) const
	{
		float sdf1;
		if(!getSDFvalue(c(0), c(1), c(2), sdf1))
			return false;
		float sdf2;
		if(!getSDFvalue(c(0), c(1), c(2) - 1, sdf2))
			sdf2 = sdf1;

		deriv = sdf1 - sdf2 / BaseClass::mResolution(2);
		return true;
	}

	float getCurvature(const Vector3i& c) const
	{
		float sdfC;
		bool res = MarchingCubes::getSDFvalue(c, sdfC);
		assert(res);

		float phiX, phiY, phiZ;
		res = getSDFderivX(c, phiX);
		assert(res);
		res = getSDFderivY(c, phiY);
		assert(res);
		res = getSDFderivZ(c, phiZ);
		assert(res);

		float phiX_, phiY_, phiZ_;
		if(!getSDFderivX(c + Eigen::Vector3i(1,0,0), phiX_))
			phiX_ = phiX;
		if(!getSDFderivY(c + Eigen::Vector3i(0,1,0), phiY_))
			phiY_ = phiY;
		if(!getSDFderivZ(c + Eigen::Vector3i(0,0,1), phiZ_))
			phiZ_ = phiZ;

		float phiXX = (phiX_ - phiX) / BaseClass::mResolution(0);
		float phiYY = (phiY_ - phiY) / BaseClass::mResolution(1);
		float phiZZ = (phiZ_ - phiZ) / BaseClass::mResolution(2);

		//compute second derivitives of different arguments (XY, XZ, YZ)
		float sdfXY_, sdfYZ_, sdfXZ_, sdfX_, sdfY_, sdfZ_;
		if(!MarchingCubes::getSDFvalue(c - Eigen::Vector3i(0,1,1), sdfYZ_))
			sdfYZ_ = sdfC;
		if(!MarchingCubes::getSDFvalue(c - Eigen::Vector3i(1,0,1), sdfXZ_))
			sdfXZ_ = sdfC;
		if(!MarchingCubes::getSDFvalue(c - Eigen::Vector3i(1,1,0), sdfXY_))
			sdfXY_ = sdfC;
		if(!MarchingCubes::getSDFvalue(c - Eigen::Vector3i(0,0,1), sdfZ_))
			sdfZ_ = sdfC;
		if(!MarchingCubes::getSDFvalue(c - Eigen::Vector3i(0,1,0), sdfY_))
			sdfY_ = sdfC;
		if(!MarchingCubes::getSDFvalue(c - Eigen::Vector3i(1,0,0), sdfX_))
			sdfX_ = sdfC;

		float phiXY, phiXZ, phiYZ;
		phiXY = (sdfC - sdfY_ - sdfX_ + sdfXY_) / (BaseClass::mResolution(0) * BaseClass::mResolution(1));
		phiXZ = (sdfC - sdfZ_ - sdfX_ + sdfXZ_) / (BaseClass::mResolution(0) * BaseClass::mResolution(2));
		phiYZ = (sdfC - sdfY_ - sdfZ_ + sdfYZ_) / (BaseClass::mResolution(1) * BaseClass::mResolution(2));
		auto grad = getSDFGrad(c);
		float gradNorm = grad.norm();
		float gradNorm3 = gradNorm * gradNorm * gradNorm;
		float curvature = (
					phiX * phiX * phiYY +
					phiY * phiY * phiXX -
					2 * phiX * phiY * phiXY +
					phiX * phiX * phiZZ +
					phiZ * phiZ * phiXX -
					2 * phiX * phiZ * phiXZ +
					phiY * phiY * phiZZ +
					phiZ * phiZ * phiYY -
					2 * phiY * phiZ * phiYZ
					) / (gradNorm3 + 1e-6);
		return curvature;
	}



	
	std::vector<Eigen::Vector3i> getNeighbourCells(Eigen::Vector3i baseCell, int kernelSize, int offset, Real depth)
	{
		std::vector<Eigen::Vector3i> neighbors;
		neighbors.reserve(kernelSize * kernelSize * kernelSize);
		neighbors.push_back(baseCell);
		Eigen::Vector3d grad = getSDFGrad(baseCell);
		if(grad.dot(grad) < 1e-6)
			return neighbors;
		grad.normalize();
		
		neighbors.reserve(kernelSize * offset * kernelSize * offset * kernelSize * offset * 8);
		for(int i = -kernelSize * offset; i <= kernelSize * offset; i += offset)
			for(int j = -kernelSize * offset; j <= kernelSize * offset; j += offset)
				for(int k = -kernelSize * offset; k <= kernelSize * offset; k += offset)
				{
					if(i == 0 && j == 0 && k == 0)
						continue;
					
					//if projection of the point offset to gradient in the point is larger, that half of the kernel - dont take the point for bluering while 
					Eigen::Vector3d offsetVector = Eigen::Vector3d(-i * BaseClass::mResolution(0),
																   -j * BaseClass::mResolution(1),
																   -k * BaseClass::mResolution(0));
					if(std::fabs(offsetVector(0) * grad(0) + offsetVector(1) * grad(1) + offsetVector(2) * grad(2)) > (depth * kernelSize * BaseClass::mResolution(0)))
						continue;
					
					neighbors.push_back(Eigen::Vector3i(baseCell(0) + i, baseCell(1) + j, baseCell(2) + k));
				}
		
		return neighbors;
	}
	
	float	mSmoothingFactor	{ 1 };
	size_t	mKernelSize			{ 1 };
	size_t	mOffset				{ 1 };
	float	mKernelDepth		{ 0.5 };
	std::unordered_map<size_t, Real> mLevelSetValues;
	bool	mBlurrSurfaceCellsOnly {false};
	size_t mBlurIterations {1};
};

typedef  BlurredReconstruction<ZhuBridsonReconstruction, std::shared_ptr<learnSPH::FluidSystem> , const Eigen::Vector3d , 
								const Eigen::Vector3d , const Eigen::Vector3d , float > ZhuBridsonBlurred;
typedef  BlurredReconstruction<SolenthilerReconstruction, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, const Eigen::Vector3d, 
								const Eigen::Vector3d , float, float, float> SolenthilerBlurred;
typedef  BlurredReconstruction<NaiveMarchingCubes, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, 
								const Eigen::Vector3d, const Eigen::Vector3d, float> NaiveBlurred;


