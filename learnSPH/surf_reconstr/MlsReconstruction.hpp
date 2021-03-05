#pragma once
#include "NaiveMarchingCubes.hpp"
#include "ZhuBridsonReconstruction.hpp"
#include "SolenthilerReconstruction.hpp"
#include <learnSPH/core/vtk_writer.h>
#include <learnSPH/core/kernel.h>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <vector>
#include <unordered_set>
#include <list>
#include <set>
#include <random>
#include <cassert>

#if 0
#ifndef DBG
#define DBG
#include <random>
#endif
#endif

//unkomment one of the variants
//#define MLSV1
//#define MLSV2
#define MLSV3

template<class BaseClass, class... Args>
class MlsReconstruction : public BaseClass
{
public:
	explicit MlsReconstruction(Args... args,
							   float smoothingFactor,
							   int iterations,
							   size_t maxSamples,
							   size_t curvatureParticles,
							   float sampleOverlapFactor):
		BaseClass(args...),
		mSmoothingFactor(smoothingFactor),
		mIterations(iterations),
		mMaxSamples(maxSamples),
		mCurvatureParticles(curvatureParticles),
		mSampleOverlapFactor(sampleOverlapFactor)
	{
		assert(smoothingFactor >= 0);
		assert(sampleOverlapFactor >= 0 && sampleOverlapFactor <= 1);

	}
	MlsReconstruction(const MlsReconstruction& other):
		BaseClass(other),
		mLevelSet(other.mLevelSet),
		mSmoothingFactor(other.mSmoothingFactor),
		mIterations(other.mIterations),
		mMaxSamples(other.mMaxSamples),
		mCurvatureParticles(other.mCurvatureParticles),
		mSampleOverlapFactor(other.mSampleOverlapFactor)
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

		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "LevelSetBeforeMls_" + MarchingCubes::mFrameNumber + ".vtk",
									 points, levelSetBeforeMls);
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "LevelSetAfterMls_" + MarchingCubes::mFrameNumber + ".vtk",
									 points, levelSetAfterMls);


#endif
	}
#ifndef MLSV1
	void correctLevelSet()
	{
		//find intersection cells
		std::unordered_map<size_t, size_t> intersectionCells = MarchingCubes::computeIntersectionVertices(0);
		std::unordered_map<size_t /*cell*/, float /*cnt*/> cellsAppearence;
		std::vector<std::vector<Eigen::Vector3i>> clusters;
		auto newLevelSet = mLevelSet;
#ifdef DBG
		std::vector<Real> intersCellsSmoothingFactor;
		std::vector<Vector3R> intersectionCellsPts;
#endif
		auto computeClusters = [this](const std::unordered_map<size_t, size_t>& intersectionCells)
		{
#ifdef MLSV2
			std::unordered_set<size_t> keys;
			keys.rehash(intersectionCells.size());
			for(const auto& item : intersectionCells)
				keys.insert(item.first);
#else
			std::vector<size_t> keys;
			keys.reserve(intersectionCells.size());
			for(const auto& item : intersectionCells)
				keys.push_back(item.first);
#endif
			std::vector<std::vector<Eigen::Vector3i>> clusters;
#ifdef MLSV2
			while(!keys.empty())
#else
			#pragma omp parallel for schedule(static)
			for(int i = 0; i < keys.size(); i++)
#endif
			{

				std::vector<Eigen::Vector3i> cluster;
#ifdef MLSV2
				auto key = *keys.begin();
#else
				auto key = keys[i];
#endif
				Real curvature; bool res = BaseClass::getCurvature(key, curvature);
				assert(res);

				auto c = BaseClass::cell(key);
				float levelSetFactor = std::min(1/std::fabs(curvature), BaseClass::mFluid->getDiameter() * mCurvatureParticles) /
														  (BaseClass::mFluid->getDiameter() * mCurvatureParticles);
				levelSetFactor *= levelSetFactor;
				assert(levelSetFactor >= 0 && levelSetFactor <= 1);
				size_t maxSamples = std::max(1, static_cast<int>(mMaxSamples * levelSetFactor));

				cluster = getNeighbourCells(intersectionCells, c, maxSamples);
				assert(cluster.size() != 0);
#ifdef MLSV2
				size_t removeParticles = cluster.size() * mSampleOverlapFactor;
				if(!removeParticles)
					removeParticles = 1;
				for(size_t i = 0; i < removeParticles; i++)
				{
					size_t nbI = BaseClass::cellIndex(cluster[i]);
					keys.erase(nbI);
				}
#endif
#pragma omp critical(UpdateClusters)
{
				clusters.push_back(std::move(cluster));
}
			}
#ifdef MLSV2
			assert(keys.empty() && "keys not empty");
#endif
			return clusters;
		};


		//compute clusters from intersection cells
		clusters = computeClusters(intersectionCells);

		//compute mls surface within cluster
		for(const auto& cluster : clusters)
		{
			auto solution = getMlsSurface(cluster.front(), cluster);
			//smooth all cluster points within the generated surface
			for(const auto& item : cluster)
			{
				auto itemI = BaseClass::cellIndex(item);
				if(!cellsAppearence.count(itemI))
				{
					newLevelSet.at(itemI) = correctSdf(solution, BaseClass::cellCoord(item));
					cellsAppearence[itemI] = 1;
				}
				else
				{
					newLevelSet.at(itemI) += correctSdf(solution, BaseClass::cellCoord(item));
					cellsAppearence.at(itemI) += 1;
				}
			}
		}
		assert(cellsAppearence.size() == intersectionCells.size());
		for(const auto& item : cellsAppearence)
		{

			//compute levelSetFactor
			float levelSetFactor = std::min(1.f, mSmoothingFactor * static_cast<float>(BaseClass::mSurfaceCells.at(item.first)) / BaseClass::mPartPerSupportArea);
			assert(levelSetFactor >= 0 && levelSetFactor <= 1);
			levelSetFactor *= levelSetFactor;
#ifdef DBG
			intersectionCellsPts.push_back(BaseClass::cellCoord(BaseClass::cell(item.first)));
			intersCellsSmoothingFactor.push_back(levelSetFactor);
#endif

			//compute new level set value as weighted sum of old oone and new one
			newLevelSet.at(item.first) /= item.second;
			newLevelSet.at(item.first) = newLevelSet.at(item.first) * levelSetFactor+ mLevelSet.at(item.first) * (1 - levelSetFactor);
		}
#ifdef DBG
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName +  "MlsSdfIntersectionVertices+SmoothingFactors_" + BaseClass::mFrameNumber + ".vtk",
									 intersectionCellsPts, intersCellsSmoothingFactor);
#endif
		mLevelSet.swap(newLevelSet);
	}
#else
	void correctLevelSet()
	{
		std::unordered_map<size_t, float> levelSet = mLevelSet;
		const std::unordered_map<size_t, size_t> surfaceCells = MarchingCubes::computeIntersectionVertices();

#ifdef DBG
		float averageNeighbors = 0;
		std::vector<double> mlsMaxSamplesPerCell;
		std::vector<double> mlsCurvature;
		std::vector<Eigen::Vector3d> points;
		static std::mt19937 generator(1);
		static std::uniform_real_distribution<float> distr {0, 1};
		static auto dice = std::bind(distr, generator);
		float probability = std::min(1., 1000. / BaseClass::mSurfaceCells.size());
		int nbCount = 0;
		std::vector<Vector3R> intersectionCells;

#endif
		for(auto cellItem : surfaceCells)
		{
			Real curvature; bool res = BaseClass::getCurvature(cellItem.first, curvature);
			assert(res);
			size_t maxSamples = std::max(static_cast<int>(1), static_cast<int>(mMaxSamples *
					std::min(1/curvature, BaseClass::mFluid->getDiameter() * 50) /
					(BaseClass::mFluid->getDiameter() * 50)));

			Eigen::Vector3i c = MarchingCubes::cell(cellItem.first);
#if 0
			std::vector<Eigen::Vector3i> nbs = getNeighbourCells(c,
																 kernelSize,
																 kernelOffset,
																 maxSamples,
																 kernelDepth);
#else
			std::vector<Eigen::Vector3i> nbs = getNeighbourCells(surfaceCells,
																 c,
																 maxSamples);
#endif
#ifdef DBG
			if(dice() < probability)
			{
				std::vector<Vector3R> nbsPts;
				nbsPts.reserve(nbs.size());
				for(auto p : nbs)
					nbsPts.push_back(BaseClass::cellCoord(p));
				learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName +  "MlsNeighbors_" + BaseClass::mFrameNumber + "_" + to_string(nbCount) + ".vtk",
											 nbsPts);
				nbCount++;
			}
			intersectionCells.push_back(BaseClass::cellCoord(c));
			averageNeighbors += nbs.size();
			mlsMaxSamplesPerCell.push_back(maxSamples);
			mlsCurvature.push_back(curvature);
			points.push_back(BaseClass::cellCoord(c));

#endif
			float newLevenSetValue = levelSet[cellItem.first];
			if(nbs.size() != 1)
			{
				auto mlsSurface = getMlsSurface(c, nbs);
				newLevenSetValue = correctSdf(mlsSurface, BaseClass::cellCoord(c));
			}


			Real smoothFactor = std::min(1.f, mSmoothingFactor * static_cast<float>(cellItem.second) / BaseClass::mPartPerSupportArea);
			smoothFactor = -1 * std::pow(1 - smoothFactor*smoothFactor, 10.) + 1;
			levelSet[cellItem.first] =  levelSet[cellItem.first] * (1 - smoothFactor) + smoothFactor * newLevenSetValue;
		}
#ifdef DBG
		averageNeighbors /= surfaceCells.size();
		std::cout << "average mls samples: " << averageNeighbors << std::endl;
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "mlsMaxSamplesPerCell_" + MarchingCubes::mFrameNumber + ".vtk",
									 points, mlsMaxSamplesPerCell);
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "mlsCurvature_" + MarchingCubes::mFrameNumber + ".vtk",
									 points, mlsCurvature);
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName +  "MlsSdfIntersectionVertices_" + BaseClass::mFrameNumber + ".vtk",
									 intersectionCells);


#endif
		mLevelSet.swap(levelSet);
	}
#endif

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
#if 0
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
#endif

	std::vector<Eigen::Vector3i> getNeighbourCells(const std::unordered_map<size_t, size_t>& cellSet,
												   const Eigen::Vector3i& baseCell,
												   int maxSamples)
	{
		std::vector<size_t> todoCells;
		size_t currentTodoCell = 0;
		std::unordered_set<size_t> todoCellsHashmap;
		std::unordered_set<size_t> ngbCells;
		std::vector<Eigen::Vector3i> ngbCellsVector;

		todoCells.reserve(maxSamples);
		todoCellsHashmap.rehash(maxSamples);
		todoCellsHashmap.max_load_factor(2);
		ngbCells.rehash(maxSamples);
		ngbCells.max_load_factor(2);
		ngbCellsVector.reserve(maxSamples);

		auto baseCellI = BaseClass::cellIndex(baseCell);
		todoCells.push_back(baseCellI);
		todoCellsHashmap.insert(baseCellI);

		while(ngbCells.size() < maxSamples && currentTodoCell < todoCells.size())
		{
			size_t currentCell = todoCells[currentTodoCell];
			currentTodoCell++;
			todoCellsHashmap.erase(currentCell);
			ngbCells.insert(currentCell);
			ngbCellsVector.push_back(BaseClass::cell(currentCell));
			Eigen::Vector3i c = BaseClass::cell(currentCell);
			for(int i = -1; i <= 1; i++)
			{
				for(int j = -1; j <= 1; j++)
				{
					for(int k = -1; k <= 1; k++)
					{
						auto cI = BaseClass::cellIndex(c + Eigen::Vector3i(i,j,k));
						if(cellSet.count(cI) && !ngbCells.count(cI) && todoCellsHashmap.count(cI) == 0)
						{
							todoCells.push_back(cI);
							todoCellsHashmap.insert(cI);
						}
					}
				}
			}
		}

		return ngbCellsVector;

	}


	Eigen::VectorXd getMlsSurface(const Eigen::Vector3i& baseCell, const std::vector<Eigen::Vector3i>& cellNeighbors)
	{
//		if(cellNeighbors.size() == 1)
//		{
//			float sdf; bool res = MarchingCubes::getSDFvalue(baseCell, sdf);
//			assert(res);
//			return sdf;
//		}

		//compute matrix B
		Eigen::MatrixXd B(cellNeighbors.size(), 5);
		float maxCellDistSqr = 0;
		auto bcC = BaseClass::cellCoord(baseCell);
		for(int i = 0; i < cellNeighbors.size(); i++)
		{
			auto cC = BaseClass::cellCoord(cellNeighbors[i]);
			B(i,0) = 1;
			B(i, 1) = cC(0);
			B(i, 2) = cC(1);
			B(i, 3) = cC(2);
			B(i, 4) = cC.dot(cC);
			float newSquaredNorm = (cC - bcC).squaredNorm();
			if(newSquaredNorm > maxCellDistSqr)
				maxCellDistSqr = newSquaredNorm;

		}

		//compute matrix W
		Eigen::MatrixXd W = Eigen::MatrixXd::Identity(cellNeighbors.size(), cellNeighbors.size());
		auto cC = BaseClass::cellCoord(baseCell);
//		auto weight = [maxCellDistSqr, &bcC](const Vector3R& cC)
//		{
//			return -(bcC - cC).squaredNorm() / maxCellDistSqr + 1;
//		};
//		for(int i = 0; i < cellNeighbors.size(); i++)
//		{
//			auto nC = BaseClass::cellCoord(cellNeighbors[i]);
//			W(i, i) = weight(nC);
//		}

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

		return solution;

	}
	float correctSdf(const Eigen::VectorXd& solution, const Eigen::Vector3d& cC)
	{
		assert(solution.size() == 5);
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
	size_t mMaxSamples {10};
	int mIterations {1};
	float mSmoothingFactor {1};
	size_t mCurvatureParticles {20};
	float mSampleOverlapFactor {0.5};
};

typedef  MlsReconstruction<ZhuBridsonReconstruction, std::shared_ptr<learnSPH::FluidSystem> , const Eigen::Vector3d ,
								const Eigen::Vector3d , const Eigen::Vector3d , float > ZhuBridsonMls;
typedef  MlsReconstruction<SolenthilerReconstruction, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, const Eigen::Vector3d,
								const Eigen::Vector3d , float, float, float> SolenthilerMls;
typedef  MlsReconstruction<NaiveMarchingCubes, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d,
								const Eigen::Vector3d, const Eigen::Vector3d, float> NaiveMls;

