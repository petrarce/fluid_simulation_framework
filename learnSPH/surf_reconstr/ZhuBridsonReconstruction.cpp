#include "ZhuBridsonReconstruction.hpp"

void ZhuBridsonReconstruction::updateGrid()
{
	dAvg.clear();
	xAvg.clear();
	denominators.clear();
	mSurfaceCells.clear();
	mPartPerSupportArea = (8 * mRadii * mRadii * mRadii) / 
							(mFluid->getDiameter() * mFluid->getDiameter() * mFluid->getDiameter());	
	const auto& particles = mFluid->getPositions();
	for(int i = 0; i < mSurfaceParticlesCount; i++)
	{
		auto nCells = getNeighbourCells(particles[i], mRadii, false);
		for(const auto& nc : nCells)
		{
			auto cI = cellIndex(nc);
			if(mSurfaceCells.find(cI) == mSurfaceCells.end())
				mSurfaceCells[cI] = 1;
			else
				mSurfaceCells[cI]++;
		}
	}
}

void ZhuBridsonReconstruction::updateLevelSet()
{
	updateDenominators();
	updateAvgs();
}

void ZhuBridsonReconstruction::configureHashTables()
{
	MarchingCubes::configureHashTables();
	xAvg.max_load_factor(mSurfaceCells.max_load_factor());
	dAvg.max_load_factor(mSurfaceCells.max_load_factor());
	denominators.max_load_factor(mSurfaceCells.max_load_factor());
}

float ZhuBridsonReconstruction::getSDFvalue(int i, int j, int k) const
{
	auto cell = Eigen::Vector3i(i,j,k);
	auto cI = cellIndex(cell);
	auto cC = cellCoord(cell);
	auto xAvgI = xAvg.find(cI);
	auto dAvgI = dAvg.find(cI);
	if(xAvgI == xAvg.end())
		//TODO compute some acceptable value
		throw std::invalid_argument("cell is not in the domain");
	
	return (cC - xAvgI->second).norm() - dAvgI->second / 2;
}

void ZhuBridsonReconstruction::updateDenominators()
{
	const vector<Vector3R>& particles = mFluid->getPositions();

	for(size_t i = 0; i < particles.size(); i++)
	{
		std::vector<Vector3i> neighbours = getNeighbourCells(particles[i], mRadii);
		for(const auto& cell : neighbours){
			auto cI = cellIndex(cell);
			auto cC = cellCoord(cell);
			auto denominatorsI = denominators.find(cI);
			if(denominatorsI == denominators.end())
				denominators[cI] = 0;
			denominators[cI] += learnSPH::kernel::kernelCubic(cC, particles[i], mRadii);
		}
	}
}

void ZhuBridsonReconstruction::updateAvgs()
{
	const std::vector<Vector3R>& particles = mFluid->getPositions();

	for(size_t i = 0; i < particles.size(); i++)
	{
		std::vector<Vector3i> neighbours = getNeighbourCells(particles[i], mRadii);
		Real minDist = std::numeric_limits<Real>::max();
		for(const auto& cell : neighbours)
		{
			auto cI = cellIndex(cell);
			auto cC = cellCoord(cell);
			auto xAvgI = xAvg.find(cI);
			auto dAvgI = dAvg.find(cI);
			if(xAvg.find(cI) == xAvg.end())
			{
				xAvg[cI] = Vector3R(0,0,0);
				dAvg[cI] = 0;
			}
			auto weight = learnSPH::kernel::kernelCubic(cC, particles[i], mRadii);
			xAvg[cI] += weight / denominators[cI] * particles[i];
			dAvg[cI] += weight / denominators[cI] * mFluid->getDiameter();
		}
	}
}

