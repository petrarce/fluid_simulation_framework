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
							   float kernelDepth,
							   float similarityThreshold,
							   float smoothingFactor,
							   bool surfaceCellsOnly,
							   int iterations):
		BaseClass(args...),
		mKernelSize(kernelSize),
		mKernelOffset(kernelOffset),
		mSdfSimilarityThreshold(similarityThreshold),
		mSurfaceCellsOnly(surfaceCellsOnly),
		mSmoothingFactor(smoothingFactor),
		mKernelDepth(kernelDepth),
		mIterations(iterations)
	{
		assert(similarityThreshold > 0);
	}
	MlsReconstruction(const MlsReconstruction& other):
		BaseClass(other),
		mLevelSet(other.mLevelSet),
		mKernelSize(other.mKernelSize),
		mKernelOffset(other.mKernelOffset),
		mSdfSimilarityThreshold(other.mSdfSimilarityThreshold),
		mSurfaceCellsOnly(other.mSurfaceCellsOnly),
		mSmoothingFactor(other.mSmoothingFactor),
		mKernelDepth(other.mKernelDepth),
		mIterations(other.mIterations)
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
		for(int i = 0; i < mIterations; i++)
			correctLevelSet();
#ifdef DBG
		for(const auto& ptItem : mLevelSet)
			levelSetAfterMls.push_back(ptItem.second);

		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "LevelSetBeforeMls" + MarchingCubes::mFrameNumber + ".vtk",
									 points, levelSetBeforeMls);
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "LevelSetAfterMls" + MarchingCubes::mFrameNumber + ".vtk",
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
		float averageNeighbors = 0;
		float averageKernelSize = 0;
		float averageKernelOffset = 0;
		std::vector<double> mlsMaxSamplesPerCell;
		std::vector<double> mlsCurvature;
		std::vector<Eigen::Vector3d> points;
#endif
		for(auto cellItem : *surfaceCells)
		{
			Real curvature; bool res = BaseClass::getCurvature(cellItem.first, curvature);
			assert(res);
			int kernelOffset = mKernelOffset;
			int kernelSize = mKernelSize;
			float kernelDepth = mKernelDepth;
			size_t maxSamples = std::max(1, static_cast<int>(std::pow(2*kernelSize + 1, 3) *
					std::min(1/curvature, BaseClass::mFluid->getDiameter() * 50) /
					(BaseClass::mFluid->getDiameter() * 50)));

			Eigen::Vector3i c = MarchingCubes::cell(cellItem.first);
			std::vector<Eigen::Vector3i> nbs = getNeighbourCells(c,
																 kernelSize,
																 kernelOffset,
																 maxSamples,
																 kernelDepth);
#ifdef DBG
			averageNeighbors += nbs.size();
			averageKernelSize += kernelSize;
			averageKernelOffset += kernelOffset;
			mlsMaxSamplesPerCell.push_back(maxSamples);
			mlsCurvature.push_back(curvature);
			points.push_back(BaseClass::cellCoord(c));

#endif
			float newLevenSetValue = getMlsCorrectedSdf(c, nbs, kernelSize, kernelOffset);

			Real smoothFactor = std::min(1.f, mSmoothingFactor * static_cast<float>(cellItem.second) / BaseClass::mPartPerSupportArea);
			smoothFactor = -1 * std::pow(1 - smoothFactor*smoothFactor, 10.) + 1;
			levelSet[cellItem.first] =  levelSet[cellItem.first] * (1 - smoothFactor) + smoothFactor * newLevenSetValue;

		}
#ifdef DBG
		averageNeighbors /= surfaceCells->size();
		averageKernelOffset /= surfaceCells->size();
		averageKernelSize /= surfaceCells->size();
		std::cout << "average mls samples: " << averageNeighbors << std::endl;
		std::cout << "average mls KernelSize: " << averageKernelSize << std::endl;
		std::cout << "average mls KernelOffset: " << averageKernelOffset << std::endl;
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "mlsMaxSamplesPerCell" + MarchingCubes::mFrameNumber + ".vtk",
									 points, mlsMaxSamplesPerCell);
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "mlsCurvature" + MarchingCubes::mFrameNumber + ".vtk",
									 points, mlsCurvature);

#endif
		if(mSurfaceCellsOnly)
			delete surfaceCells;
		mLevelSet.swap(levelSet);
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

	std::vector<Eigen::Vector3i> getNeighbourCells(const Eigen::Vector3i& baseCell,
												   int kernelSize,
												   int kernelOffset,
												   int maxSamples,
												   float depth) const
	{
		assert(depth >= 0 && depth <= 1);
		std::vector<Eigen::Vector3i> neighbors;
		neighbors.reserve(pow(kernelSize, 3));
		float baseSdf; bool res = MarchingCubes::getSDFvalue(baseCell, baseSdf);
		assert(res);
		Eigen::Vector3d grad = getSDFGrad(baseCell);
		neighbors.push_back(baseCell);
		if(grad.dot(grad) < 1e-6)
			return neighbors;
		int cnt = 1;

		auto pushNeighbor = [
				&cnt,
				&grad,
				&baseCell,
				depth,
				kernelSize,
				kernelOffset,
				this,
				baseSdf,
				&neighbors](const Eigen::Vector3i& newNb)
		{
			cnt++;
			int i = newNb(0);
			int j = newNb(1);
			int k = newNb(2);

			if(i == 0 && j == 0 && k == 0)
				return;

			//if projection of the point offset to gradient in the point is larger, that half of the kernel - dont take the point for bluering while
			Eigen::Vector3d offsetVector = Eigen::Vector3d(-i * BaseClass::mResolution(0),
														   -j * BaseClass::mResolution(1),
														   -k * BaseClass::mResolution(2));
			if(std::fabs(offsetVector(0) * grad(0) + offsetVector(1) * grad(1) + offsetVector(2) * grad(2)) > (depth * kernelSize * BaseClass::mResolution(0)))
				return;


			Eigen::Vector3i c = baseCell + Eigen::Vector3i(i * kernelOffset, j * kernelOffset, k * kernelOffset);
			float sdf; bool res = MarchingCubes::getSDFvalue(c, sdf);
			if(!res || std::fabs(sdf - baseSdf) > mSdfSimilarityThreshold)
				return;
			neighbors.push_back(c);
		};

		int level = 1;
		//traverce levels
		while(
#ifndef DBG
			  neighbors.size() < maxSamples &&
#endif
			  level <= kernelSize)
		{
			pushNeighbor({0, 0, level});
			pushNeighbor({0, 0, -level});
			pushNeighbor({0, level, 0});
			pushNeighbor({0, -level, 0});
			pushNeighbor({level, 0, 0});
			pushNeighbor({-level, 0, 0});

			//traverce faces
			for(int i = 1; i <= level-1; i++)
			{
				pushNeighbor({0, i, level});
				pushNeighbor({0, -i, level});
				pushNeighbor({i, 0, level});
				pushNeighbor({-i, 0, level});

				pushNeighbor({0, i, -level});
				pushNeighbor({0, -i, -level});
				pushNeighbor({i, 0, -level});
				pushNeighbor({-i, 0, -level});

				pushNeighbor({0,  level, i});
				pushNeighbor({0,  level, -i});
				pushNeighbor({i,  level, 0});
				pushNeighbor({-i,  level, 0});

				pushNeighbor({0,  -level, i});
				pushNeighbor({0,  -level, -i});
				pushNeighbor({i,  -level, 0});
				pushNeighbor({-i,  -level, 0});

				pushNeighbor({level, 0, i});
				pushNeighbor({level, 0, -i});
				pushNeighbor({level, i, 0});
				pushNeighbor({level, -i, 0});

				pushNeighbor({-level, 0, i});
				pushNeighbor({-level, 0, -i});
				pushNeighbor({-level, i, 0});
				pushNeighbor({-level, -i, 0});
				//traverse face edges
				for(int j = 1; j <= i - 1; j++)
				{
					pushNeighbor({j, i, level});
					pushNeighbor({j, -i, level});
					pushNeighbor({i, -j, level});
					pushNeighbor({-i, -j, level});

					pushNeighbor({j, i, -level});
					pushNeighbor({j, -i, -level});
					pushNeighbor({i, -j, -level});
					pushNeighbor({-i, -j, -level});

					pushNeighbor({j,  level, i});
					pushNeighbor({j,  level, -i});
					pushNeighbor({i,  level, -j});
					pushNeighbor({-i,  level, -j});

					pushNeighbor({j,  -level, i});
					pushNeighbor({j,  -level, -i});
					pushNeighbor({i,  -level, -j});
					pushNeighbor({-i,  -level, -j});

					pushNeighbor({level, j, i});
					pushNeighbor({level, j, -i});
					pushNeighbor({level, i, -j});
					pushNeighbor({level, -i, -j});

					pushNeighbor({-level, j, i});
					pushNeighbor({-level, j, -i});
					pushNeighbor({-level, i, -j});
					pushNeighbor({-level, -i, -j});
				}
				//traverce face corners
				pushNeighbor({i, i, level});
				pushNeighbor({-i, -i, level});
				pushNeighbor({i, -i, level});
				pushNeighbor({-i, i, level});

				pushNeighbor({i, i, -level});
				pushNeighbor({-i, -i, -level});
				pushNeighbor({i, -i, -level});
				pushNeighbor({-i, i, -level});

				pushNeighbor({i,  level, i});
				pushNeighbor({-i,  level, -i});
				pushNeighbor({i,  level, -i});
				pushNeighbor({-i,  level, i});

				pushNeighbor({i,  -level, i});
				pushNeighbor({-i,  -level, -i});
				pushNeighbor({i,  -level, -i});
				pushNeighbor({-i,  -level, i});

				pushNeighbor({level, i, i});
				pushNeighbor({level, -i, -i});
				pushNeighbor({level, i, -i});
				pushNeighbor({level, -i, i});

				pushNeighbor({-level, i, i});
				pushNeighbor({-level, -i, -i});
				pushNeighbor({-level, i, -i});
				pushNeighbor({-level, -i, i});

			}


			//traverse cube edges
			for(int j = 1; j <= level - 1; j++)
			{
				pushNeighbor({j, level, level});
				pushNeighbor({-j, level, level});
				pushNeighbor({j, -level, level});
				pushNeighbor({-j, -level, level});
				pushNeighbor({j, level, -level});
				pushNeighbor({-j, level, -level});
				pushNeighbor({j, -level, -level});
				pushNeighbor({-j, -level, -level});

				pushNeighbor({level, -j, level});
				pushNeighbor({level, j, level});
				pushNeighbor({level, -j, -level});
				pushNeighbor({level, j, -level});
				pushNeighbor({level, -j, level});
				pushNeighbor({level, j, level});
				pushNeighbor({level, -j, -level});
				pushNeighbor({level, j, -level});

				pushNeighbor({-level, -j, level});
				pushNeighbor({-level, j, level});
				pushNeighbor({-level, -j, -level});
				pushNeighbor({-level, j, -level});
				pushNeighbor({-level, -j, level});
				pushNeighbor({-level, j, level});
				pushNeighbor({-level, -j, -level});
				pushNeighbor({-level, j, -level});
			}
			//add Corner Centers
			pushNeighbor({0, level, level});
			pushNeighbor({0, level, -level});
			pushNeighbor({0, -level, level});
			pushNeighbor({0, -level, -level});
			pushNeighbor({-level, 0, level});
			pushNeighbor({-level, 0, -level});
			pushNeighbor({level, 0, level});
			pushNeighbor({level, 0, -level});
			pushNeighbor({level, level, 0});
			pushNeighbor({level, -level, 0});
			pushNeighbor({-level, level, 0});
			pushNeighbor({-level, -level, 0});

			//traverse corners
			pushNeighbor({level, level, level});
			pushNeighbor({level, level, -level});
			pushNeighbor({level, -level, level});
			pushNeighbor({level, -level, -level});
			pushNeighbor({-level, level, level});
			pushNeighbor({-level, level, -level});
			pushNeighbor({-level, -level, level});
			pushNeighbor({-level, -level, -level});

			level++;
		}

#ifdef DBG
		assert(cnt == pow(2 * kernelSize + 1, 3));
#endif
		neighbors.resize(std::min(static_cast<int>(neighbors.size()), maxSamples));
		return neighbors;
	}

	float getMlsCorrectedSdf(const Eigen::Vector3i& baseCell, const std::vector<Eigen::Vector3i>& cellNeighbors,
							 int kernelSize,
							 int kernelOffset)
	{
		if(cellNeighbors.size() == 1)
		{
			float sdf; bool res = MarchingCubes::getSDFvalue(baseCell, sdf);
			assert(res);
			return sdf;
		}

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
			W(i, i) = learnSPH::kernel::kernelFunction(cC, nC, 1.5 * BaseClass::mResolution(0) * (kernelSize * kernelOffset) + 1e-6);
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
//	int mMaxNeighborNodes {-1};
	float mKernelDepth {1};
	int mIterations {1};
	float mSmoothingFactor {1};
};

typedef  MlsReconstruction<ZhuBridsonReconstruction, std::shared_ptr<learnSPH::FluidSystem> , const Eigen::Vector3d ,
								const Eigen::Vector3d , const Eigen::Vector3d , float > ZhuBridsonMls;
typedef  MlsReconstruction<SolenthilerReconstruction, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, const Eigen::Vector3d,
								const Eigen::Vector3d , float, float, float> SolenthilerMls;
typedef  MlsReconstruction<NaiveMarchingCubes, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d,
								const Eigen::Vector3d, const Eigen::Vector3d, float> NaiveMls;

