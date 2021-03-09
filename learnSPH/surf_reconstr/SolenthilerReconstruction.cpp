#include "SolenthilerReconstruction.hpp"
#include <learnSPH/core/vtk_writer.h>
#if 0
#ifndef DBG
#define DBG
#endif
#endif

void SolenthilerReconstruction::updateLevelSet()
{
	ZhuBridsonReconstruction::updateDenominators();
	ZhuBridsonReconstruction::updateAvgs();
	updateGradientSums();

	updateJakobians();
	denominators.clear();
	denominators.shrink_to_fit();
	mGradSums.clear();
	mGradSums.shrink_to_fit();

	updateFFunction();
	mJacobians.clear();
	mJacobians.shrink_to_fit();

	mMcVertexSdf.clear();
	mMcVertexSdf.resize(mDataToCellIndex.size(), 0);
	for(size_t i = 0; i < mMcVertexSdf.size(); i++)
	{
		//	auto cell = Eigen::Vector3li(i,j,k);
		//	auto cI = cellIndex(cell);
		//	auto cC = cellCoord(cell);
		//	auto xAvgI = xAvg.find(cI);
		//	auto dAvgI = dAvg.find(cI);
		//	auto fVal = mCellGradComponents.find(cI);
		//	if(xAvgI == xAvg.end())
		//		//TODO compute some acceptable value
		//		return false;
		//	sdf = ((cC - xAvgI->second).norm() - dAvgI->second * fVal->second.fVal / 2);
		//	return true;

		DataIndex dI(i, *this);
		assert(*dI != InvPrt);
		auto cC = cellCoord(cell(*dI));
		mMcVertexSdf[i] = ((cC - xAvg[i]).norm() - dAvg[i] * mFVal[i] / 2);
	}
	xAvg.clear();
	xAvg.shrink_to_fit();
	dAvg.clear();
	dAvg.shrink_to_fit();
	mFVal.clear();
	mFVal.shrink_to_fit();
}

void SolenthilerReconstruction::updateGrid()
{
	ZhuBridsonReconstruction::updateGrid();
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
	mGradSums.clear();
	mGradSums.resize(mDataToCellIndex.size(), Vector3R::Zero());
	const auto& particles = mFluid->getPositions();
	//compute gradient sums
	for(size_t i = 0; i < this->mSurfaceParticlesCount; i++)
	{
		auto nCells = getNeighbourCells(particles[i], mRadii);
		for(const auto& c : nCells)
		{
			CellIndex cI(cellIndex(c), *this);
			assert(*cI != InvPrt);
			auto gW = learnSPH::kernel::kernelCubicGrad(cellCoord(c), particles[i], mRadii);
			mGradSums[*cI] += gW;
		}
	}
#ifdef DBG
	std::vector<Vector3R> gradSums;
	std::vector<Vector3R> points;
	for(const auto& item : mCellToDataIndex)
	{
		CellIndex cI(item.first, *this);
		assert(*cI != InvPrt);
		points.push_back(cellCoord(cell(cI())));
		gradSums.push_back(mGradSums[*cI]);
	}
	
	static int cnt = 0;
	learnSPH::saveParticlesToVTK("/tmp/GradSums" + to_string(cnt) + ".vtk", points, gradSums);
	cnt++;
#endif
	
}

void SolenthilerReconstruction::updateJakobians()
{
	mJacobians.clear();
	mJacobians.resize(mDataToCellIndex.size(), Matrix3d::Zero());
	const auto& particles = mFluid->getPositions();
	//compute gradient sums
	for(size_t i = 0; i < this->mSurfaceParticlesCount; i++)
	{
		auto nCells = getNeighbourCells(particles[i], mRadii);
		for(const auto& c : nCells)
		{
			CellIndex cI(cellIndex(c), *this);
			assert(*cI != InvPrt);

			auto gW = learnSPH::kernel::kernelCubicGrad(cellCoord(c), particles[i], mRadii);
			auto w = learnSPH::kernel::kernelCubic(cellCoord(c), particles[i], mRadii);
			Eigen::Matrix3d a = (particles[i] * gW.transpose())	/ denominators[*cI];
			Eigen::Matrix3d b = (particles[i] * mGradSums[*cI].transpose() * w) / (denominators[*cI] * denominators[*cI]);
			mJacobians[*cI] += a - b;
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
	mFVal.clear();
	mFVal.resize(mDataToCellIndex.size(), 0);
	assert(mFVal.size() == mJacobians.size());
	EigenSolver<Matrix3d> es;
	for(size_t i = 0; i < mFVal.size(); i++)
	{
		es.compute(mJacobians[i], false);
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
		mFVal[i] = thresholdFunction(evaluesReal(0), this->mTLow, this->mTHigh);
#ifdef DBG
		DataIndex dI(i, *this);
		assert(*dI != InvPrt);
		evls.push_back(evaluesReal);
		pts.push_back(cellCoord(cell(*dI)));
		fvls.push_back(mFVal[i]);
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
