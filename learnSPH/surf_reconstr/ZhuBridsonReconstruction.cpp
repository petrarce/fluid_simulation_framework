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
				size_t dataIndex = mDataToCellIndex.size();
				mCellToDataIndex[cI()] = dataIndex;
				mDataToCellIndex[dataIndex] = cI();
			}
		}
	}
	assert(mDataToCellIndex.size() == mCellToDataIndex.size());

	relocateFluidParticles(mRadii);
}

void ZhuBridsonReconstruction::updateLevelSet()
{
	mMcVertexSdf.clear();
	mMcVertexSdf.resize(mDataToCellIndex.size(), 0);
	updateAvgs();
	assert(xAvg.size() == mMcVertexSdf.size());
	#pragma omp parallel for schedule(static)
	for(size_t i = 0; i < mMcVertexSdf.size(); i++)
	{
		const DataIndex dI(i, *this);
		const CellIndex cI(*dI, *this);
		assert(cI() != InvPrt);
		const auto cC = cellCoord(cell(cI()));

		const float norm = (cC - xAvg[i] / denominators[i]).norm();
		mMcVertexSdf[i] = norm - mRadii / 4;
	}
	xAvg.clear();
	xAvg.shrink_to_fit();
}


void ZhuBridsonReconstruction::updateAvgs()
{

	const std::vector<Vector3R>& particles = mFluid->getPositions();
	xAvg = std::vector<Vector3R>(mDataToCellIndex.size(), Vector3R(0,0,0));
	denominators = std::vector<Real>(mDataToCellIndex.size(), 0);
	mMcVertexCurvature = std::vector<Real>(mDataToCellIndex.size(), 0);
	mMcVertexSphParticles = std::vector<size_t>(mDataToCellIndex.size(), 0);

	assert(xAvg.size() == denominators.size());
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
			#pragma omp critical(UpdateAVGS)
			xAvg[*cI] += weight * particles[i];


			#pragma omp atomic update
			denominators[*cI] += weight;

			#pragma omp critical
			mMcVertexCurvature[*cI] += mCurvature[i];

			#pragma omp critical
			mMcVertexSphParticles[*cI]++;

		}
	}
}

