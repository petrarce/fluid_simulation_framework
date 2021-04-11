#include "OnderikEtAlReconstruction.hpp"
#include <vector>
#include <Eigen/Dense>
#include <learnSPH/core/vtk_writer.h>

//#define DBG
void OnderikEtAlReconstruction::updateWeightedSums()
{
	mWeightedSums = std::vector<float>(mDataToCellIndex.size(), 0);
	mWeightedAvgs = std::vector<Vector3R>(mDataToCellIndex.size(), Eigen::Vector3d(0, 0, 0));
	mMcVertexCurvature = std::vector<Real>(mDataToCellIndex.size(), 0);
	mMcVertexSphParticles = std::vector<size_t>(mDataToCellIndex.size(), 0);

	const auto& particles = mFluid->getPositions();
	const auto& densities = mFluid->getDensities();
#pragma omp parallel for schedule(static)
	for(size_t i = 0; i < particles.size(); i++)
	{
		auto nbc = getNeighbourCells(particles[i], mRadii);
		float normalisedDencity = mFluid->getMass() / densities[i];
		for(const auto& nc : nbc)
		{
			CellIndex cI(cellIndex(nc), *this);
			auto cC = cellCoord(nc);
			Real weight = learnSPH::kernel::kernelFunction(particles[i], cC, mRadii);

			#pragma omp atomic update
			mWeightedSums[*cI] += normalisedDencity * weight;

			#pragma omp critical
			mWeightedAvgs[*cI] += normalisedDencity * weight * particles[i];

			#pragma omp atomic update
			mMcVertexSphParticles[*cI]++;

			#pragma omp atomic update
			mMcVertexCurvature[*cI] += mCurvature[i];
		}
	}
}

float OnderikEtAlReconstruction::g(float w)
{
	float n = w - mWmax;
	float d = mWmax - mWmin;

	float r = 1 - (n*n) / (d*d);
	return std::min(1.f, r*r);
}

void OnderikEtAlReconstruction::updateSdf()
{
	mMcVertexSdf = std::vector<Real>(mDataToCellIndex.size(), 0);
	assert(mMcVertexSdf.size() == mWeightedAvgs.size() &&
		   mMcVertexSdf.size() == mWeightedSums.size());
#ifdef DBG
	std::vector<Vector3R> points(mDataToCellIndex.size(), Vector3R(0,0,0));
	std::vector<double> distDecayFunction(mDataToCellIndex.size(), 0);
#endif

#pragma omp parallel for schedule(static)
	for(size_t i = 0; i < mMcVertexSdf.size(); i++)
	{
		DataIndex dI(i, *this);
		assert(*dI != InvPrt);
		auto cC = cellCoord(cell(*dI));
		mMcVertexSdf[i] = (cC - mWeightedAvgs[i] / mWeightedSums[i]).norm() - (mRadii) * g(mWeightedSums[i]);
#ifdef DBG
		points[i] = cC;
		distDecayFunction[i] = g(mWeightedSums[i]);
#endif
	}
#ifdef DBG
		std::vector<double> weightedSums(mWeightedSums.begin(), mWeightedSums.end());
		learnSPH::saveParticlesToVTK("/tmp/" + mSimName + "_WeightedSums_" + mFrameNumber + ".vtk", points, weightedSums);
		learnSPH::saveParticlesToVTK("/tmp/" + mSimName + "_DistanceDecay_" + mFrameNumber + ".vtk", points, distDecayFunction);
#endif
}
void OnderikEtAlReconstruction::updateLevelSet()
{
	updateWeightedSums();
	updateSdf();
}

