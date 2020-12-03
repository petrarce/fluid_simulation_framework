#pragma once
#include "NaiveMarchingCubes.hpp"
#include "ZhuBridsonReconstruction.hpp"
#include "SolenthilerReconstruction.hpp"
#include <learnSPH/core/vtk_writer.h>
#include <learnSPH/core/kernel.h>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <vector>

#if 0
#ifndef DBG
#define DBG
#endif
#endif

template<class BaseClass, class... Args>
class MlsReconstruction : public BaseClass
{
public:
	explicit MlsReconstruction(Args... args,
							   size_t kernelSize,
							   size_t kernelOffset,
							   float similarityThreshold,
							   bool surfaceCellsOnly):
		BaseClass(args...),
		mKernelSize(kernelSize),
		mKernelOffset(kernelOffset),
		mSdfSimilarityThreshold(similarityThreshold),
		mSurfaceCellsOnly(surfaceCellsOnly)
	{
		assert(similarityThreshold > 0);
	}
	MlsReconstruction(const MlsReconstruction& other):
		BaseClass(other),
		mLevelSet(other.mLevelSet),
		mKernelSize(other.mKernelSize),
		mKernelOffset(other.mKernelOffset),
		mSdfSimilarityThreshold(other.mSdfSimilarityThreshold),
		mSurfaceCellsOnly(other.mSurfaceCellsOnly)
	{}
private:

	void configureHashTables() override
	{
		BaseClass::configureHashTables();
		mLevelSet.max_load_factor(MarchingCubes::mSurfaceCells.max_load_factor());
		mLevelSet.rehash(MarchingCubes::mSurfaceCells.bucket_count());
	}
	void updateGrid() override
	{
		BaseClass::updateGrid();
		mLevelSet.clear();
	}

	void updateLevelSet() override
	{
		BaseClass::updateLevelSet();
		saveLevelSet();
#ifdef DBG
		std::vector<Vector3R> points;
		std::vector<Real> levelSetBeforeMls;
		std::vector<Real> levelSetAfterMls;

		for(const auto& ptItem : mLevelSet)
		{
			points.push_back(MarchingCubes::cellCoord(MarchingCubes::cell(ptItem.first)));
			levelSetBeforeMls.push_back(ptItem.second);
		}
#endif
		correctLevelSet();
#ifdef DBG
		for(const auto& ptItem : mLevelSet)
			levelSetAfterMls.push_back(ptItem.second);

		learnSPH::saveParticlesToVTK("/tmp/LevelSetBeforeMls" + MarchingCubes::mFrameNumber + ".vtk",
									 points, levelSetBeforeMls);
		learnSPH::saveParticlesToVTK("/tmp/LevelSetAfterMls" + MarchingCubes::mFrameNumber + ".vtk",
									 points, levelSetAfterMls);

#endif
	}

	void correctLevelSet()
	{
		std::unordered_map<size_t, float> levelSet = mLevelSet;
		const std::unordered_map<size_t, size_t>* surfaceCells;
		if(mSurfaceCellsOnly)
			surfaceCells = new std::unordered_map<size_t, size_t>(MarchingCubes::computeIntersectionCellVertices(mKernelSize * mKernelOffset));
		else
			surfaceCells = &(MarchingCubes::mSurfaceCells);

#ifdef DBG
		float averageNeighbors;
#endif
		for(auto cellItem : *surfaceCells)
		{
			Eigen::Vector3i c = MarchingCubes::cell(cellItem.first);
			std::vector<Eigen::Vector3i> nbs = getNeighbourCells(c, mKernelSize);
#ifdef DBG
			averageNeighbors += nbs.size();
#endif
			float newLevenSetValue = getMlsCorrectedSdf(c, nbs);
//			Real smoothFactor = std::min(1.f, 2 * static_cast<float>(cellItem.second) / BaseClass::mPartPerSupportArea);
//			smoothFactor = -1 * std::pow(1 - smoothFactor*smoothFactor, 10.) + 1;
//			levelSet[cellItem.first] = smoothFactor * newLevenSetValue + (1 - smoothFactor) * mLevelSet[cellItem.first];
			levelSet[cellItem.first] = newLevenSetValue;

		}
#ifdef DBG
		averageNeighbors /= mLevelSet.size();
		std::cout << "average mls neighbors: " << averageNeighbors << std::endl;
#endif
		if(mSurfaceCellsOnly)
			delete surfaceCells;
		mLevelSet.swap(levelSet);
	}

	std::vector<Eigen::Vector3i> getNeighbourCells(const Eigen::Vector3i& baseCell, int kernelSize) const
	{
		std::vector<Eigen::Vector3i> neighbors;
		neighbors.reserve(pow(kernelSize, 3));
		float baseSdf; bool res = MarchingCubes::getSDFvalue(baseCell, baseSdf);
		assert(res);
		for(int i = -kernelSize; i <= kernelSize; i++)
		{
			for(int j = -kernelSize; j <= kernelSize; j++)
			{
				for(int k = -kernelSize; k <= kernelSize; k++)
				{
					Eigen::Vector3i c = baseCell + Eigen::Vector3i(i * mKernelOffset, j * mKernelOffset, k * mKernelOffset);
					size_t cI = MarchingCubes::cellIndex(c);
					unordered_map<size_t, size_t>::const_iterator item = MarchingCubes::mSurfaceCells.find(cI);
					if(item == MarchingCubes::mSurfaceCells.cend())
						continue;
					float sdf;
					res = MarchingCubes::getSDFvalue(c, sdf);
					if(std::fabs(sdf - baseSdf) > mSdfSimilarityThreshold)
						continue;
					neighbors.push_back(c);
				}
			}
		}
		return neighbors;
	}

	float getMlsCorrectedSdf(const Eigen::Vector3i& baseCell, const std::vector<Eigen::Vector3i>& cellNeighbors)
	{

		//compute matrix B
		Eigen::MatrixXd B(cellNeighbors.size(), 5);
		for(int i = 0; i < cellNeighbors.size(); i++)
		{
			auto cC = BaseClass::cellCoord(cellNeighbors[i]);
			B(i,0) = 1;
			B(i, 1) = cC(0);
			B(i, 2) = cC(1);
			B(i, 3) = cC(2);
			B(i, 4) = cC.dot(cC);
		}

		//compute matrix W
		Eigen::MatrixXd W = Eigen::MatrixXd::Identity(cellNeighbors.size(), cellNeighbors.size());
		auto cC = BaseClass::cellCoord(baseCell);
		for(int i = 0; i < cellNeighbors.size(); i++)
		{
			auto nC = BaseClass::cellCoord(cellNeighbors[i]);
			//W(i,i) = learnSPH::kernel::kernelFunction(cC, nC, BaseClass::mResolution(0) * (mKernelSize) + 1e-6);
			//W(i,i) = learnSPH::kernel::kernelCubic(cC, nC, BaseClass::mResolution(0) * (mKernelSize) + 1e-6);
		}

		//compute vector u from existing SDF values
		Eigen::VectorXd u(cellNeighbors.size());
		for(int i = 0; i < cellNeighbors.size(); i++)
		{
			float sdf; bool res = MarchingCubes::getSDFvalue(cellNeighbors[i], sdf);
			assert(res);
			u(i) = sdf;
		}

		//compute A=B`WB  b=B`Wu
		auto Btmp = B.transpose() * W;
		auto A = Btmp * B;
		auto b = Btmp * u;

		//find solution of c = A^-1 b
		Eigen::VectorXd solution = A.colPivHouseholderQr().solve(b);
		assert(solution.rows() == 5);

		Eigen::VectorXd indecis(5);
		indecis(0) = 1;
		indecis(1) = cC(0);
		indecis(2) = cC(1);
		indecis(3) = cC(2);
		indecis(4) = cC.dot(cC);

		return indecis.dot(solution);
	}

	void saveLevelSet()
	{
		for(const auto& cItem : BaseClass::mSurfaceCells)
		{
			float sdf;
			auto c = MarchingCubes::cell(cItem.first);
			bool res = BaseClass::getSDFvalue(c(0), c(1), c(2), sdf);
			assert(res);
			mLevelSet[cItem.first] = sdf;
		}
	}

	bool getSDFvalue(int i, int j, int k, float& sdf) const override
	{
		auto cI = MarchingCubes::cellIndex(Eigen::Vector3i(i, j, k));
		auto item = mLevelSet.find(cI);
		if( item == mLevelSet.end())
			return false;
		sdf = item->second;
		return true;
	}

private:
	unordered_map<size_t, float> mLevelSet;
	int mKernelSize {1};
	int mKernelOffset {1};
	float mSdfSimilarityThreshold {0.5};
	bool mSurfaceCellsOnly {false};
};

typedef  MlsReconstruction<ZhuBridsonReconstruction, std::shared_ptr<learnSPH::FluidSystem> , const Eigen::Vector3d ,
								const Eigen::Vector3d , const Eigen::Vector3d , float > ZhuBridsonMls;
typedef  MlsReconstruction<SolenthilerReconstruction, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, const Eigen::Vector3d,
								const Eigen::Vector3d , float, float, float> SolenthilerMls;
typedef  MlsReconstruction<NaiveMarchingCubes, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d,
								const Eigen::Vector3d, const Eigen::Vector3d, float> NaiveMls;

