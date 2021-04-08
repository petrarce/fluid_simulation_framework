#pragma once
#include "NaiveMarchingCubes.hpp"
#include "ZhuBridsonReconstruction.hpp"
#include "SolenthilerReconstruction.hpp"
#include <learnSPH/core/vtk_writer.h>
#include <learnSPH/core/PerfStats.hpp>
#include <learnSPH/core/kernel.h>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <list>
#include <set>
#include <random>
#include <cassert>
#include <iostream>
#define USE_QUADRADIC_SAMPLING 1

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
							   size_t mlsSamples,
							   size_t maxSamples,
							   size_t curvatureParticles,
							   float sampleOverlapFactor,
							   float clusterFraction):
		BaseClass(args...),
		mSmoothingFactor(smoothingFactor),
		mIterations(iterations),
		mMlsSamples(mlsSamples),
		mMaxSamples(maxSamples),
		mCurvatureParticles(curvatureParticles),
		mSampleOverlapFactor(sampleOverlapFactor),
		mCluestersFraction(clusterFraction)
	{
		assert(smoothingFactor >= 0);
		assert(sampleOverlapFactor >= 0 && sampleOverlapFactor <= 1);
		assert(clusterFraction > 0 && clusterFraction <= 1);

	}
	MlsReconstruction(const MlsReconstruction& other):
		BaseClass(other),
		mSmoothingFactor(other.mSmoothingFactor),
		mIterations(other.mIterations),
		mMlsSamples(other.mMlsSamples),
		mMaxSamples(other.mMaxSamples),
		mCurvatureParticles(other.mCurvatureParticles),
		mSampleOverlapFactor(other.mSampleOverlapFactor),
		mCluestersFraction(other.mCluestersFraction)
	{}
private:

	void updateGrid() override
	{
		BaseClass::updateGrid();
	}

	std::vector<size_t> generateIndexes(std::function<float(float)> shiftFunction)
	{
		std::mt19937 gen(1996);
		std::unordered_set<size_t> acceptedIndices;
		std::vector<size_t> acceptedIndicesVector;
		acceptedIndicesVector.reserve(mMaxSamples);

		acceptedIndices.insert(0);
		acceptedIndicesVector.push_back(0);
		size_t lowerBound = 0;
		size_t upperBound = (mMlsSamples - 1);
		for(int i = 0; i < std::min(mMaxSamples, mMlsSamples); i++)
		{
			Real value = static_cast<Real>(gen()) / gen.max();
			size_t index = lowerBound + shiftFunction(value) * (upperBound - lowerBound);				size_t currentIndex = index;
			while(acceptedIndices.count(currentIndex)){
				if(currentIndex == lowerBound)
					currentIndex = upperBound;
				else
					currentIndex--;
				assert(currentIndex != index);
			}
			acceptedIndices.insert(currentIndex);
			acceptedIndicesVector.push_back(currentIndex);
			if(currentIndex == lowerBound)
			{
				currentIndex++;
				while(acceptedIndices.count(currentIndex))
				{
					currentIndex++;
					assert(currentIndex < upperBound);
				}
				lowerBound = currentIndex;
			}
			else if(currentIndex == upperBound)
			{
				currentIndex--;
				while(acceptedIndices.count(currentIndex))
				{
					currentIndex--;
					assert(currentIndex > lowerBound);
				}
				upperBound = currentIndex;
			}
			assert(upperBound > lowerBound);
		}
		std::sort(acceptedIndicesVector.begin(), acceptedIndicesVector.end());
		return acceptedIndicesVector;

	}

	void updateLevelSet() override
	{
		BaseClass::updateLevelSet();
#ifdef DBG
		std::vector<Vector3R> points;

		for(size_t i = 0; i < BaseClass::mMcVertexSdf.size(); i++)
		{
			typename BaseClass::DataIndex dI(i, *this);
			assert(*dI != InvPrt);
			points.push_back(MarchingCubes::cellCoord(MarchingCubes::cell(*dI)));
		}
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "LevelSetBeforeMls_" + MarchingCubes::mFrameNumber + ".vtk",
									 points, BaseClass::mMcVertexSdf);
#endif
		globalPerfStats.startTimer("mlsSmoothPath");
		for(int i = 0; i < mIterations; i++)
			correctLevelSet();
		globalPerfStats.stopTimer("mlsSmoothPath");

#ifdef DBG

		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName + "LevelSetAfterMls_" + MarchingCubes::mFrameNumber + ".vtk",
									 points, BaseClass::mMcVertexSdf);

#endif
	}

	std::vector<std::vector<size_t>> computeClusters(const std::unordered_set<size_t>& intersectionCells, std::vector<size_t>& todoIntersectionCells)
	{

#ifdef DBG
		std::vector<Vector3R> intPts;
		std::vector<Real> maxFluidPartFactor;
		std::vector<Real> curvaturePts;
#endif
		std::vector<size_t>& keys = todoIntersectionCells;

		std::vector<size_t> acceptedIndices;
		if(mMlsSamples > 2 * mMaxSamples)
		{
#if USE_QUADRADIC_SAMPLING
			auto fn = [](float v) {return v*v;};
#else
			auto fn = [](float v) {return v;};
#endif
			acceptedIndices = generateIndexes(fn);
		}
		else if(mMlsSamples > mMaxSamples)
		{
			assert(mMlsSamples/static_cast<float>(mMaxSamples) > 1);
			for(float i = 0; i < mMaxSamples; i+= mMlsSamples/static_cast<float>(mMaxSamples))
				acceptedIndices.push_back(static_cast<size_t>(i));
		}

		std::vector<std::vector<size_t>> clusters;
		size_t clustersSize = std::min(20000ul, todoIntersectionCells.size());
		#pragma omp parallel for schedule(static)
		for(size_t i = keys.size() - 1; i > keys.size() - clustersSize; i--)
		{

			std::vector<size_t> cluster;
			auto key = keys[i];
			typename BaseClass::CellIndex cI(key, *this);
			Real curvature; bool res = BaseClass::getCurvature(cI, curvature);
			assert(res);

			auto c = BaseClass::cell(cI());
//			float levelSetFactor = std::min(1.f/std::fabs(curvature), BaseClass::mFluid->getDiameter() * mCurvatureParticles) /
//													  (BaseClass::mFluid->getDiameter() * mCurvatureParticles);
//			levelSetFactor *= levelSetFactor;
//			assert(levelSetFactor >= 0 && levelSetFactor <= 1);
//			size_t maxSamples = std::max(1, static_cast<int>(mMaxSamples * levelSetFactor));
			size_t maxSamples = mMlsSamples;

			cluster = getNeighbourCells(intersectionCells, c, maxSamples, acceptedIndices);
			assert(cluster.size() != 0);
			#pragma omp critical(UpdateClusters)
			{
			clusters.push_back(std::move(cluster));
#ifdef DBG

			intPts.push_back(BaseClass::cellCoord(BaseClass::cell(cI())));
//			maxFluidPartFactor.push_back(levelSetFactor);
			curvaturePts.push_back(curvature);
#endif
			}
		}
#ifdef DBG
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName +  "_MlsMaxSamplesFactor_" + BaseClass::mFrameNumber + ".vtk",
									 intPts, maxFluidPartFactor);
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName +  "_MlsIntersectionCellsSCurvature_" + BaseClass::mFrameNumber + ".vtk",
									 intPts, curvaturePts);

#endif
		assert(todoIntersectionCells.size() >= clustersSize);
		todoIntersectionCells.resize(todoIntersectionCells.size() - clustersSize);
		return clusters;
	};
#ifndef MLSV1
	void correctLevelSet()
	{
		//find intersection cells
		std::unordered_set<size_t> intersectionCells = MarchingCubes::computeIntersectionVertices(0);
		std::vector<std::vector<size_t>> clusters;
		std::vector<float /*total weight*/> cellsAppearence(BaseClass::mMcVertexSdf.size(), 0);
		std::vector<Real> newLevelSet = BaseClass::mMcVertexSdf;
#ifdef DBG
		std::vector<Real> intersCellsSmoothingFactor;
		std::vector<Vector3R> intersectionCellsPts;
#endif


#ifdef DBG
		std::vector<Real> mlsIntersectionCellsSdfBefore;
		std::vector<Real> mlsIntersectionCellsSdfAfter;

#endif

		//compute clusters from intersection cells
		std::vector<size_t> todoIntersectionCells(intersectionCells.begin(), intersectionCells.end());
		std::vector<size_t> tmp;
		tmp.reserve(todoIntersectionCells.size());
		size_t clusterSamples = mCluestersFraction * todoIntersectionCells.size();
		std::mt19937 gen(123);
		for(size_t i = 0; i < clusterSamples; i++)
		{
			size_t index = gen() % todoIntersectionCells.size();
			tmp.push_back(todoIntersectionCells[index]);
			todoIntersectionCells[index] = todoIntersectionCells.back();
			todoIntersectionCells.pop_back();
		}
		todoIntersectionCells.swap(tmp);

		while(!todoIntersectionCells.empty())
		{
			clusters = computeClusters(intersectionCells, todoIntersectionCells);
#ifdef DBG
			int clusterCnt = 0;
			const int printEveryCluster = clusters.size() / 100;
#endif
			//compute mls surface within cluster
			#pragma omp parallel for schedule(static)
			for(size_t i = 0; i < clusters.size(); i++)
			{
				const auto& cluster = clusters[i];

	#ifdef DBG
				std::vector<Vector3R> clusterPts;
				float error = 0;
	#endif
				auto solution = getMlsSurface(MarchingCubes::cell(cluster.front()), cluster);
				Real maxDist = (MarchingCubes::cellCoord(MarchingCubes::cell(cluster.front())) -
						MarchingCubes::cellCoord(MarchingCubes::cell(cluster.back()))).norm();
				//smooth all cluster points within the generated surface
				for(const auto& item : cluster)
				{
					MarchingCubes::CellIndex cI(item, *this);
					Real dist = (MarchingCubes::cellCoord(MarchingCubes::cell(cluster.front())) -
								 MarchingCubes::cellCoord(MarchingCubes::cell(item))).norm();
					Real weight = std::clamp((maxDist - dist) / maxDist, 0., 1.);
					assert(weight >= 0 && weight <= 1);
					assert(*cI != InvPrt);
					#pragma omp atomic update
					newLevelSet[*cI] += weight * correctSdf(solution, BaseClass::cellCoord(MarchingCubes::cell(cI())));

					#pragma omp atomic update
					cellsAppearence[*cI] += weight;
	#ifdef DBG
					#pragma omp critical(DBGPushclusterPts)
					clusterPts.push_back(BaseClass::cellCoord(MarchingCubes::cell(cI())));


	#endif
				}
	#ifdef DBG
				#pragma omp critical(DBGPrintCluster)
				{
				if((clusterCnt % printEveryCluster) == 0)
				{
					std::vector<Real> ids(clusterPts.size(), -1);
					ids[0] = 1;
					learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName +  "MlsCluster" + BaseClass::mFrameNumber
												 + "_" + to_string(clusterCnt) + ".vtk",
												 clusterPts, ids);

				}
				clusterCnt++;
				}
	#endif

			}
		}
#ifdef DBG
		float totalError = 0;
#endif
		#pragma omp parallel for schedule(static)
		for(size_t i = 0; i < cellsAppearence.size(); i++)
		{
			assert(cellsAppearence[i] >= 0);
			if(cellsAppearence[i] < 1e-9)
				continue;
			//compute levelSetFactor
			float levelSetFactor = std::min(1.f, mSmoothingFactor * static_cast<float>(BaseClass::mMcVertexSphParticles[i]) /
					BaseClass::mPartPerSupportArea);
			assert(levelSetFactor >= 0 && levelSetFactor <= 1);
			levelSetFactor *= levelSetFactor;
#ifdef DBG
			#pragma omp critical
			{
			MarchingCubes::DataIndex dI(i, *this);
			if(intersectionCells.count(*dI))
			{
				intersectionCellsPts.push_back(BaseClass::cellCoord(BaseClass::cell(*dI)));
				intersCellsSmoothingFactor.push_back(levelSetFactor);
			}
			}

#endif

			//compute new level set value as weighted sum of old oone and new one
			newLevelSet[i] -= BaseClass::mMcVertexSdf[i];
			newLevelSet[i] /= cellsAppearence[i];
			newLevelSet[i] = newLevelSet[i] * levelSetFactor + BaseClass::mMcVertexSdf[i] * (1 - levelSetFactor);

#ifdef DBG
			float newPart = (newLevelSet[i] - BaseClass::mMcVertexSdf[i]);
			#pragma omp atomic update
			totalError += (newPart * newPart);

			#pragma omp critical
			{
			mlsIntersectionCellsSdfBefore.push_back(BaseClass::mMcVertexSdf[i]);
			mlsIntersectionCellsSdfAfter.push_back(newLevelSet[i]);
			}
#endif
		}
#ifdef DBG
		pr_info("mls standard deviation: %f", totalError);

		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName +  "MlsSdfIntersectionVertices+SmoothingFactors_" + BaseClass::mFrameNumber + ".vtk",
									 intersectionCellsPts, intersCellsSmoothingFactor);
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName +  "MlsIntersectionVellsSdfBefore_" + BaseClass::mFrameNumber + ".vtk",
									 intersectionCellsPts, mlsIntersectionCellsSdfBefore);
		learnSPH::saveParticlesToVTK("/tmp/" + BaseClass::mSimName +  "MlsIntersectionVellsSdfAfter_" + BaseClass::mFrameNumber + ".vtk",
									 intersectionCellsPts, mlsIntersectionCellsSdfAfter);



#endif
		BaseClass::mMcVertexSdf.swap(newLevelSet);
	}
#else
#endif

	Eigen::Vector3d getSDFGrad(const Vector3li& c) const
	{
		float cellSDFval;
		bool res = MarchingCubes::getSDFvalue(c, cellSDFval);
		assert(res);
		float sdfValX = 0.f;
		float sdfValY = 0.f;
		float sdfValZ = 0.f;
		if(!MarchingCubes::getSDFvalue(c - Vector3li(1, 0, 0), sdfValX))
			sdfValX = cellSDFval;
		if(!MarchingCubes::getSDFvalue(c - Vector3li(0, 1, 0), sdfValY))
			sdfValY = cellSDFval;
		if(!MarchingCubes::getSDFvalue(c - Vector3li(0, 0, 1), sdfValZ))
			sdfValZ = cellSDFval;
		float dx = (cellSDFval - sdfValX) / BaseClass::mResolution(0);
		float dy = (cellSDFval - sdfValY) / BaseClass::mResolution(1);
		float dz = (cellSDFval - sdfValZ) / BaseClass::mResolution(2);

		return Eigen::Vector3d(dx, dy, dz);
	}


	std::vector<size_t> getNeighbourCells(const std::unordered_set<size_t>& cellSet,
												   const Eigen::Vector3li& baseCell,
												   int maxSamples,
													const std::vector<size_t>& acceptedIndices)
	{
		std::vector<size_t> todoCells;
		size_t currentTodoCell = 0;
		std::unordered_set<size_t> todoCellsHashmap;
		std::unordered_set<size_t> ngbCells;
		std::vector<size_t> ngbCellsVector;

		todoCells.reserve(maxSamples);
		todoCellsHashmap.rehash(maxSamples);
		todoCellsHashmap.max_load_factor(2);
		ngbCells.rehash(maxSamples);
		ngbCells.max_load_factor(2);
		ngbCellsVector.reserve(maxSamples);

		auto baseCellI = BaseClass::cellIndex(baseCell);
		todoCells.push_back(baseCellI);
		todoCellsHashmap.insert(baseCellI);
		size_t currentAcceptedIndex = 0;

		while(ngbCells.size() < maxSamples && currentTodoCell < todoCells.size())
		{
			size_t currentCell = todoCells[currentTodoCell];
			todoCellsHashmap.erase(currentCell);
			ngbCells.insert(currentCell);
			if(acceptedIndices.empty() || acceptedIndices[currentAcceptedIndex] == currentTodoCell)
			{
				ngbCellsVector.push_back(currentCell);
				currentAcceptedIndex++;
			}
			currentTodoCell++;
			Eigen::Vector3li c = BaseClass::cell(currentCell);
			for(int64_t i = -1; i <= 1; i++)
			{
				for(int64_t j = -1; j <= 1; j++)
				{
					for(int64_t k = -1; k <= 1; k++)
					{
						auto cI = BaseClass::cellIndex(c + Eigen::Vector3li(i,j,k));
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


	Eigen::VectorXd getMlsSurface(const Eigen::Vector3li& baseCell, const std::vector<size_t>& cellNeighbors)
	{
		//compute matrix B
		Eigen::MatrixXd B(cellNeighbors.size(), 5);
		float maxCellDistSqr = 0;
		auto bcC = BaseClass::cellCoord(baseCell);
		for(size_t i = 0; i < cellNeighbors.size(); i++)
		{
			auto cC = BaseClass::cellCoord(MarchingCubes::cell(cellNeighbors[i]));
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

		//compute vector u from existing SDF values
		Eigen::VectorXd u(cellNeighbors.size());
		for(size_t i = 0; i < cellNeighbors.size(); i++)
		{
			MarchingCubes::CellIndex cI(cellNeighbors[i], *this);
			float sdf; bool res = MarchingCubes::getSDFvalue(cI, sdf);
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

private:
	size_t mMlsSamples {100};
	size_t mMaxSamples {100};
	int mIterations {1};
	float mSmoothingFactor {1};
	size_t mCurvatureParticles {20};
	float mSampleOverlapFactor {0.5};
	float mCluestersFraction {0.5};
};

typedef  MlsReconstruction<ZhuBridsonReconstruction, std::shared_ptr<learnSPH::FluidSystem> , const Eigen::Vector3d ,
								const Eigen::Vector3d , const Eigen::Vector3d , float > ZhuBridsonMls;
typedef  MlsReconstruction<SolenthilerReconstruction, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, const Eigen::Vector3d,
								const Eigen::Vector3d , float, float, float> SolenthilerMls;
typedef  MlsReconstruction<NaiveMarchingCubes, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d,
								const Eigen::Vector3d, const Eigen::Vector3d, float> NaiveMls;

