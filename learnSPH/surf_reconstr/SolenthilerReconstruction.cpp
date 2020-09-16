#include "SolenthilerReconstruction.hpp"
#include <learnSPH/core/vtk_writer.h>

void SolenthilerReconstruction::updateLevelSet()
{
	ZhuBridsonReconstruction::updateLevelSet();
	updateGradientSums();
	updateJakobians();
	updateFFunction();
}

void SolenthilerReconstruction::updateGrid()
{
	ZhuBridsonReconstruction::updateGrid();
	mCellGradComponents.clear();
	for(const auto& surfCell : mSurfaceCells)
		mCellGradComponents[surfCell.first] = {Matrix3d::Zero(), 0, Vector3R::Zero()};
}

void SolenthilerReconstruction::configureHashTables()
{
	ZhuBridsonReconstruction::configureHashTables();
	mCellGradComponents.max_load_factor(mSurfaceCells.max_load_factor());
}

static float thresholdFunction(Real EVmax, Real tMin, Real tMax)
{
	assert(tMax > tMin);
	if(EVmax < tMin)
		return 1.;
	
	Real lmbda = (tMax - EVmax) / (tMax - tMin);
	Real lmbda2 = lmbda * lmbda;
	return std::max(0., lmbda2 * lmbda - 3 * lmbda2 + 3*lmbda);
}

void SolenthilerReconstruction::updateGradientSums()
{
	const auto& particles = mFluid->getPositions();
	//compute gradient sums
	for(size_t i = 0; i < this->mSurfaceParticlesCount; i++)
	{
		auto nCells = getNeighbourCells(particles[i], mRadii);
		for(const auto& c : nCells)
		{
			auto gW = learnSPH::kernel::kernelCubicGrad(cellCoord(c), particles[i], mRadii);
			mCellGradComponents[cellIndex(c)].gradSum += gW;
		}
	}
}

void SolenthilerReconstruction::updateJakobians()
{
	const auto& particles = mFluid->getPositions();
	//compute gradient sums
	for(size_t i = 0; i < this->mSurfaceParticlesCount; i++)
	{
		auto nCells = getNeighbourCells(particles[i], mRadii);
		for(const auto& c : nCells)
		{
			size_t cI = cellIndex(c);
			auto gW = learnSPH::kernel::kernelCubicGrad(cellCoord(c), particles[i], mRadii);
			auto w = learnSPH::kernel::kernelCubic(cellCoord(c), particles[i], mRadii);
			mCellGradComponents[cI].jakobian += (particles[i] * 
				gW.transpose())	/ denominators[cI];
			mCellGradComponents[cI].jakobian -= (particles[i] * mCellGradComponents[cI].gradSum.transpose() * w) / 
					(denominators[cI] * denominators[cI]);
		}
	}
}
void SolenthilerReconstruction::updateFFunction()
{
#ifdef DEBUG
	std::vector<Real> evls;
	std::vector<Vector3R> pts;
	std::vector<Real> fvls;
#endif
	
	//compute fFunction values
	EigenSolver<Matrix3d> es;
	for(auto& f : mCellGradComponents)
	{
		es.compute(f.second.jakobian, false);
		if(es.info() != Eigen::ComputationInfo::Success)
		{
			pr_dbg("eigenvalue evaluation error");
			assert(es.info() == Eigen::ComputationInfo::Success);
		}
		auto evalues = es.eigenvalues();
		Vector3R evaluesReal = Vector3R(/*std::fabs*/(evalues(0).real()), 
										/*std::fabs*/(evalues(1).real()), 
										/*std::fabs*/(evalues(2).real()));
		std::sort(evaluesReal.data(), evaluesReal.data() + evaluesReal.size(), [](Real v1, Real v2) {return v1 > v2; });
		f.second.fVal = thresholdFunction(evaluesReal(0), this->mTLow, this->mTHigh);
#ifdef DEBUG
		evls.push_back(evaluesReal(0));
		pts.push_back(cellCoord(cell(mSurfaceCells[f.first])));
		fvls.push_back(f.second.fVal);
#endif
	}
#ifdef DEBUG
	static int cnt = 0;
	learnSPH::saveParticlesToVTK("/tmp/EVALUES" + to_string(cnt) + ".vtk", 
								 pts, evls);
	learnSPH::saveParticlesToVTK("/tmp/FVALUES" + to_string(cnt) + ".vtk", 
								 pts, fvls);

	cnt++;
#endif
	
}
float SolenthilerReconstruction::getSDFvalue(int i, int j, int k) const
{
	auto cell = Eigen::Vector3i(i,j,k);
	auto cI = cellIndex(cell);
	auto cC = cellCoord(cell);
	auto xAvgI = xAvg.find(cI);
	auto rAvgI = rAvg.find(cI);
	auto fVal = mCellGradComponents.find(cI);
	if(xAvgI == xAvg.end())
		//TODO compute some acceptable value
		return -1;
	return -1 * ((cC - xAvgI->second).norm() - rAvgI->second * fVal->second.fVal);
}
