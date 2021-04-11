#pragma once
#include "NaiveMarchingCubes.hpp"

class MullerEtAlReconstruction: public NaiveMarchingCubes
{
public:
	explicit MullerEtAlReconstruction(std::shared_ptr<learnSPH::FluidSystem> fluid,
		const Eigen::Vector3d lCorner,
		const Eigen::Vector3d uCorner,
		const Eigen::Vector3d cResolution,
		float initValue
	):
		NaiveMarchingCubes(fluid, lCorner, uCorner, cResolution, initValue)
	{
	}
private:
	void updateLevelSet() override
	{
		if(!mFluid)
			throw std::runtime_error("fluid was not initialised");

		mMcVertexSdf.clear();
		mMcVertexCurvature.clear();
		mMcVertexSphParticles.clear();
		mMcVertexSdf.resize(mDataToCellIndex.size(), mInitialValue);
		mMcVertexCurvature.resize(mDataToCellIndex.size(), 0);
		mMcVertexSphParticles.resize(mDataToCellIndex.size(), 0);


		const vector<Vector3R>& positions = mFluid->getPositions();
		const vector<Real>& densities = mFluid->getDensities();

		#pragma omp parallel for schedule(static)
		for(size_t i = 0; i < mFluid->size(); i++)
		{
			double fluidDensity = densities[i];
			const Eigen::Vector3d& particle = positions[i];
			std::vector<Eigen::Vector3li> neighbourCells =
					getNeighbourCells(particle, mFluid->getCompactSupport());
			for(const auto& cell : neighbourCells)
			{
				CellIndex cI(cellIndex(cell), *this);
				assert(*cI != static_cast<size_t>(-1));
				float weight = learnSPH::kernel::kernelFunction(
					particle,
					cellCoord(cell),
					mFluid->getSmoothingLength()
				);
				float density = fluidDensity;
				float total = mFluid->getMass() / density * weight;
				#pragma omp atomic update
				mMcVertexSdf[*cI] += total;

				#pragma omp atomic update
				mMcVertexCurvature[*cI] += mCurvature[i];

				#pragma omp atomic update
				mMcVertexSphParticles[*cI]++;
			}
		}
		//no need in particle curvature any more. Clear storage
		mCurvature.clear();
	}
};
