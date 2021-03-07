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
	mCellGradComponents.rehash(mSurfaceCells.bucket_count());
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
#ifdef DBG
	std::vector<Vector3R> gradSums;
	std::vector<Vector3R> points;
	for(const auto& item : mSurfaceCells)
	{
		points.push_back(cellCoord(cell(item.first)));
		gradSums.push_back(mCellGradComponents[item.first].gradSum);
	}
	
	static int cnt = 0;
	learnSPH::saveParticlesToVTK("/tmp/GradSums" + to_string(cnt) + ".vtk", points, gradSums);
	cnt++;
#endif
	
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
			Eigen::Matrix3d a = (particles[i] * gW.transpose())	/ denominators[cI];
			Eigen::Matrix3d b = (particles[i] * mCellGradComponents[cI].gradSum.transpose() * w) / (denominators[cI] * denominators[cI]);
			mCellGradComponents[cI].jakobian += a - b;
		}
	}
}
void SolenthilerReconstruction::updateFFunction()
{
#ifdef DBG
	std::vector<Eigen::Vector3d> evls;
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
		std::sort(evaluesReal.data(), evaluesReal.data() + evaluesReal.size(), [](Real v1, Real v2) { return v1 > v2; });
		f.second.fVal = thresholdFunction(evaluesReal(0), this->mTLow, this->mTHigh);
#ifdef DBG
		evls.push_back(evaluesReal);
		pts.push_back(cellCoord(cell(mSurfaceCells[f.first])));
		fvls.push_back(f.second.fVal);
#endif
	}
#ifdef DBG
	static int cnt = 0;
	learnSPH::saveParticlesToVTK("/tmp/EVALUES" + to_string(cnt) + ".vtk", 
								 pts, evls);
	learnSPH::saveParticlesToVTK("/tmp/FVALUES" + to_string(cnt) + ".vtk", 
								 pts, fvls);

	cnt++;
#endif
	
}
bool SolenthilerReconstruction::getSDFvalue(size_t i, size_t j, size_t k, float& sdf) const
{
	auto cell = Eigen::Vector3li(i,j,k);
	auto cI = cellIndex(cell);
	auto cC = cellCoord(cell);
	auto xAvgI = xAvg.find(cI);
	auto dAvgI = dAvg.find(cI);
	auto fVal = mCellGradComponents.find(cI);
	if(xAvgI == xAvg.end())
		//TODO compute some acceptable value
		return false;
	sdf = ((cC - xAvgI->second).norm() - dAvgI->second * fVal->second.fVal / 2);
	return true;
}
