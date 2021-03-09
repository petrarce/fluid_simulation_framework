#include "ZhuBridsonReconstruction.hpp"

#if 0
#ifndef DBG
#define DBG
#endif
#endif

void ZhuBridsonReconstruction::updateGrid()
{
	mDataToCellIndex.clear();
	mCellToDataIndex.clear();
	mMcVertexCurvature.clear();
	mMcVertexCurvature.reserve(mSurfaceParticlesCount);
	mMcVertexSphParticles.reserve(mSurfaceParticlesCount);
	mPartPerSupportArea = (8 * mRadii * mRadii * mRadii) /
							(mFluid->getDiameter() * mFluid->getDiameter() * mFluid->getDiameter());	
	const auto& particles = mFluid->getPositions();
	for(size_t i = 0; i < mSurfaceParticlesCount; i++)
	{
		auto nCells = getNeighbourCells(particles[i], mRadii, false);
		for(const auto& nc : nCells)
		{
			CellIndex cI(cellIndex(nc), *this);
			if(*cI == InvPrt)
			{
				mCellToDataIndex[cI()] = mMcVertexSphParticles.size();
				mDataToCellIndex[mMcVertexSphParticles.size()] = cI();
				mMcVertexSphParticles.push_back(1);
				mMcVertexCurvature.push_back(mCurvature[i]);
			}
			else
			{
				mMcVertexSphParticles[*cI]++;
				mMcVertexCurvature[*cI] += mCurvature[i];
			}
		}
	}
	assert(mDataToCellIndex.size() == mCellToDataIndex.size());
	mMcVertexCurvature.shrink_to_fit();
	mMcVertexSphParticles.shrink_to_fit();
}

void ZhuBridsonReconstruction::updateLevelSet()
{
	mMcVertexSdf.clear();
	mMcVertexSdf.resize(mDataToCellIndex.size(), 0);
	updateDenominators();
	updateAvgs();
	denominators.clear();
	denominators.shrink_to_fit();
	assert(xAvg.size() == dAvg.size() && xAvg.size() == mMcVertexSdf.size());
#pragma omp parallel for schedule(static)
	for(size_t i = 0; i < xAvg.size(); i++)
	{
			//		-       auto cell = Eigen::Vector3li(i,j,k);
			//		-       auto cI = cellIndex(cell);
			//		-       auto cC = cellCoord(cell);
			//		-       auto xAvgI = xAvg.find(cI);
			//		-       auto dAvgI = dAvg.find(cI);
			//		-       if(xAvgI == xAvg.end())
			//		-               //TODO compute some acceptable value
			//		-               return false;
			//		-
			//		-       float norm = (cC - xAvgI->second).norm();
			//		-       float radiusAvg = dAvgI->second;
			//		-       sdf = norm - radiusAvg / 2;
			//		-       return true;
		const DataIndex dI(i, *this);
		const CellIndex cI(*dI, *this);
		assert(cI() != InvPrt);
		const auto cC = cellCoord(cell(cI()));
		const float norm = (cC - xAvg[i]).norm();
		mMcVertexSdf[i] = norm - dAvg[i] / 2;
	}
	xAvg.clear();
	xAvg.shrink_to_fit();
	dAvg.clear();
	dAvg.shrink_to_fit();
}

void ZhuBridsonReconstruction::updateDenominators()
{
	denominators.clear();
	denominators.resize(mDataToCellIndex.size(), 0);
	const vector<Vector3R>& particles = mFluid->getPositions();

#pragma omp parallel for schedule(static)
	for(size_t i = 0; i < particles.size(); i++)
	{
		std::vector<Vector3li> neighbours = getNeighbourCells(particles[i], mRadii);
		for(const auto& cell : neighbours){
			CellIndex cI(cellIndex(cell), *this);
			assert(*cI != InvPrt && *cI < denominators.size());
			auto cC = cellCoord(cell);
#pragma omp critical(UpdateDenominators)
			{
			denominators[*cI] += learnSPH::kernel::kernelCubic(cC, particles[i], mRadii);
			}
		}
	}
}

void ZhuBridsonReconstruction::updateAvgs()
{
	const std::vector<Vector3R>& particles = mFluid->getPositions();
	dAvg.clear();
	dAvg.resize(mDataToCellIndex.size(), 0);
	xAvg.clear();
	xAvg.resize(mDataToCellIndex.size(), Vector3R(0,0,0));
	assert(xAvg.size() == denominators.size() && dAvg.size() == denominators.size());
#pragma omp parallel for schedule(static)
	for(size_t i = 0; i < particles.size(); i++)
	{
		std::vector<Vector3li> neighbours = getNeighbourCells(particles[i], mRadii);
		for(const auto& cell : neighbours)
		{
			CellIndex cI(cellIndex(cell), *this);
			assert(*cI != InvPrt && *cI < xAvg.size());
			auto cC = cellCoord(cell);
			auto weight = learnSPH::kernel::kernelCubic(cC, particles[i], mRadii);
			float denominator = denominators[*cI] + 1e-6;
#pragma omp critical(UpdateAVGS)
			{
			xAvg[*cI] += weight / denominator * particles[i];
			dAvg[*cI] += weight / denominator * mFluid->getDiameter();
			}
		}
	}
}

