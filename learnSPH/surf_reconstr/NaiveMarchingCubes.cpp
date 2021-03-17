#include <memory>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <learnSPH/core/storage.h>
#include <learnSPH/core/kernel.h>
#include <learnSPH/core/vtk_writer.h>
#include <learnSPH/core/PerfStats.hpp>
#include <learnSPH/simulation/solver.h>
#include "look_up_tables.hpp"
#include "NaiveMarchingCubes.hpp"
#include <unordered_set>

#if 0
#ifndef DBG
#define DBG
#endif
#endif
using namespace learnSPH;
MarchingCubes::MarchingCubes(std::shared_ptr<learnSPH::FluidSystem> fluid,
								const Eigen::Vector3d lCorner,
								const Eigen::Vector3d uCorner,
								const Eigen::Vector3d cResolution,
								float initValue):
	  mFluid(fluid)
	, mLowerCorner(lCorner)
	, mUpperCorner(uCorner)
	, mDimentions(static_cast<size_t>(std::fabs((lCorner(0) - uCorner(0)) / cResolution(0))),
		static_cast<size_t>(std::fabs((lCorner(1) - uCorner(1)) / cResolution(1))),
		static_cast<size_t>(std::fabs((lCorner(2) - uCorner(2)) / cResolution(2))))
	, mResolution(cResolution)
	, mInitialValue(initValue)
{
}

void MarchingCubes::configureHashTables()
{
	if(mSurfaceParticlesCount)
	{
		mDataToCellIndex.max_load_factor(4);
		mCellToDataIndex.max_load_factor(4);
	}
	mDataToCellIndex.rehash(mSurfaceParticlesCount / 2);
	mCellToDataIndex.rehash(mSurfaceParticlesCount / 2);
}

std::vector<Eigen::Vector3d> MarchingCubes::generateMesh(const std::shared_ptr<learnSPH::FluidSystem> fluid)
{
	setFluidSystem(fluid);

	std::cout << "\rstage 1/5"; cout.flush();
	globalPerfStats.startTimer("updateSurfaceParticles");
	updateSurfaceParticles();
	globalPerfStats.stopTimer("updateSurfaceParticles");
#ifdef DBG
	saveParticlesToVTK("/tmp/" + mSimName + "_ActiveFluidParticles_" + mFrameNumber + ".vtk",
					   mFluid->getPositions(), mFluid->getDensities());
#endif

	std::cout << "\rstage 2/5"; cout.flush();
	globalPerfStats.startTimer("configureHashTables");
	configureHashTables();
	globalPerfStats.stopTimer("configureHashTables");

	std::cout << "\rstage 3/5"; cout.flush();
	globalPerfStats.startTimer("updateGrid");
	updateGrid();
	globalPerfStats.stopTimer("updateGrid");

	std::cout << "\rstage 4/5"; cout.flush();
	globalPerfStats.startTimer("updateLevelSet");
	updateLevelSet();
	globalPerfStats.stopTimer("updateLevelSet");
#ifdef DBG
	vector<Real> sdf;
	vector<Vector3R> vertices;
	vector<Real> cellCurvature;
	vector<Real> particleConcentration;
	for(const auto& vert : mCellToDataIndex)
	{
		auto c = cell(vert.first);
		CellIndex cI(cellIndex(c), *this);
		auto cC = cellCoord(c);
		float sdfV; 
		bool res = getSDFvalue(c(0), c(1), c(2), sdfV);
		assert(res);
		vertices.push_back(cC);
		sdf.push_back(sdfV);
		Real curvature; getCurvature(cI, curvature);
		cellCurvature.push_back(curvature);
		particleConcentration.push_back(static_cast<Real>(vert.second) / mPartPerSupportArea);
	}
	saveParticlesToVTK("/tmp/" + mSimName + "SDF_" + mFrameNumber + ".vtk", vertices, sdf);
	saveParticlesToVTK("/tmp/" + mSimName + "CellsCurvature_" + mFrameNumber + ".vtk", vertices, cellCurvature);
	saveParticlesToVTK("/tmp/" + mSimName + "ParticleConcentrationPerSupportVolume_" + mFrameNumber + ".vtk", vertices, particleConcentration);

	auto intersectionCellVertices = computeIntersectionCellVertices();
	vector<Vector3R> intersectionCellVerticePoints;
	intersectionCellVerticePoints.reserve(intersectionCellVertices.size());
	vector<Real> intersectionCellCurvature;
	for(const auto& c : intersectionCellVertices)
	{
		CellIndex cI(c, *this);
		assert(*cI != InvPrt);
		intersectionCellVerticePoints.push_back(cellCoord(cell(cI())));
		Real curvature; getCurvature(cI, curvature);
		intersectionCellCurvature.push_back(curvature);
	}
	saveParticlesToVTK("/tmp/" + mSimName + "IntersectionCellsCurvature_" + mFrameNumber + ".vtk", intersectionCellVerticePoints, intersectionCellCurvature);

#endif
	std::cout << "\rstage 5/5"; cout.flush();
	globalPerfStats.startTimer("getTriangles");
	auto mesh = getTriangles();
	globalPerfStats.stopTimer("getTriangles");

	//cleanup all buffers
	return mesh;
}
	
void MarchingCubes::relocateFluidParticles(Real supportRadius)
{
	auto& particles = mFluid->getPositions();
	auto& densities = mFluid->getDensities();

	for(size_t i = mSurfaceParticlesCount; i < particles.size();)
	{
		auto nCells = getNeighbourCells(particles[i], supportRadius);
		if(nCells.empty())
		{
			particles[i].swap(particles.back());
			densities[i] = densities.back();
			particles.pop_back();
			densities.pop_back();
			continue;
		}
		i++;
	}
}
	
void NaiveMarchingCubes::updateGrid()
{
	mDataToCellIndex.clear();
	mCellToDataIndex.clear();
	mPartPerSupportArea = 8 * (mFluid->getSmoothingLength() * mFluid->getSmoothingLength() * mFluid->getSmoothingLength()) /
							(mFluid->getDiameter() * mFluid->getDiameter() * mFluid->getDiameter());
	const auto& particles = mFluid->getPositions();
	size_t totalCells = 0;
	for(size_t i = 0; i < mSurfaceParticlesCount; i++)
	{
		auto nCells = getNeighbourCells(particles[i], mFluid->getCompactSupport(), false);
		for(const auto& nc : nCells)
		{
			size_t cI = cellIndex(nc);
			if(!mCellToDataIndex.count(cI))
			{
				mCellToDataIndex[cI] = totalCells;
				mDataToCellIndex[totalCells] = cI;
				totalCells++;
			}
		}
	}
	assert(mDataToCellIndex.size() == mCellToDataIndex.size());
	//remove all fluid particles that are not in the neighborhood of the surface cells
	relocateFluidParticles(mFluid->getCompactSupport());
}


void NaiveMarchingCubes::updateLevelSet()
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
			float density = std::max(fluidDensity, mFluid->getRestDensity());
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

vector<Eigen::Vector3d> MarchingCubes::getTriangles() const
{

	vector<Eigen::Vector3d> triangleMesh;

	auto intersectionCells = computeIntersectionCells();
	triangleMesh.reserve(intersectionCells.size() * 3 * 3);

	std::vector<Eigen::Vector3d> triangle;
	triangle.reserve(3);
	#pragma omp parallel for schedule(static) private(triangle)
	for(size_t g = 0; g < intersectionCells.size(); g++)
	{
		const auto& cellVert = intersectionCells[g];
		Eigen::Vector3li cellInd = cell(cellVert.first);
		size_t i = cellInd(0);
		size_t j = cellInd(1);
		size_t k = cellInd(2);
		
		const std::array<std::array<int, 3>, 5>& triangle_type = cellVert.second;

		for(size_t l = 0; l < 5; l++) {
			triangle.clear();
			for(size_t m = 0; m < 3; m++) {
				if(triangle_type[l][m] == -1) break;
				
				int c1Xind = (i + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][0]);
				int c1Yind = (j + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][1]);
				int c1Zind = (k + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][2]);

				int c2Xind = (i + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][0]);
				int c2Yind = (j + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][1]);
				int c2Zind = (k + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][2]);
				float v1, v2;
				
				if(!getSDFvalue(c1Xind, c1Yind, c1Zind, v1))
				{
					bool res = getSDFvalue(i,j,k, v1);
					assert(res);
				}
				
				if(!getSDFvalue(c2Xind, c2Yind, c2Zind, v2))
				{
					bool res = getSDFvalue(i,j,k, v2);
					assert(res);
				}

				Eigen::Vector3d p1 = cellCoord(Eigen::Vector3li(c1Xind, c1Yind, c1Zind));
				Eigen::Vector3d p2 = cellCoord(Eigen::Vector3li(c2Xind, c2Yind, c2Zind));
				triangle.push_back(lerp(p1, p2, v1, v2, 0));
			}
			if(triangle.size())
			{
				#pragma omp critical(UpdateTriangleMesh)
				triangleMesh.insert(triangleMesh.end(), triangle.begin(), triangle.end());
			}
		}
	}
	return triangleMesh;
}

std::unordered_set<size_t> MarchingCubes::computeIntersectionCellVertices(int neighborsCnt) const
{
	std::unordered_set<size_t> intersectionCellVertices;
	auto intersectionCells = computeIntersectionCells();
	intersectionCellVertices.reserve(intersectionCells.size());
	for(const auto& iCell : intersectionCells)
	{
		Eigen::Vector3li c = cell(iCell.first);
		for(int i = 0; i < 8; i++)
		{
			Eigen::Vector3li nc = c + Eigen::Vector3li(1<<i & 0x4, 1 << i & 0x2, 1 << i & 0x1);
			for(int i = -neighborsCnt; i <= neighborsCnt; i++)
			{
				for(int j = -neighborsCnt; j <= neighborsCnt; j++)
				{
					for(int k = -neighborsCnt; k <= neighborsCnt; k++)
					{

						CellIndex ncI(cellIndex(nc + Eigen::Vector3li(i, j, k)), *this);
						if(intersectionCellVertices.find(ncI()) != intersectionCellVertices.end())
							continue;
						if(*ncI ==InvPrt)
							continue;
						intersectionCellVertices.insert(ncI());
					}
				}
			}
		}
	}
	return intersectionCellVertices;
}

std::unordered_set<size_t> MarchingCubes::computeIntersectionVertices(int neighbors) const
{
	std::unordered_set<size_t> intersectionVertices;
	for(auto vert : mCellToDataIndex)
	{
		CellIndex cI(vert.first, *this);
		float sdf; bool res = getSDFvalue(cI, sdf);
		assert(res);
		auto c = cell(vert.first);
		for(int i = -1; i <= 1; i++)
			for(int j = -1; j <= 1; j++)
				for(int k = -1; k <= 1; k++)
				{
					CellIndex ncI(cellIndex(c + Eigen::Vector3li(i,j,k)), *this);
					float nbSdf; res = getSDFvalue(ncI, nbSdf);
					if(!res)
						continue;
					if(nbSdf * sdf < 0)
					{
						intersectionVertices.insert(ncI());
						if(neighbors > 0)
						{
							for(int l = -neighbors; l <= neighbors; l++)
								for(int m = -neighbors; m <= neighbors; m++)
									for(int n = -neighbors; n <= neighbors; n++)
									{
										CellIndex nbI(cellIndex(c + Eigen::Vector3li(l, m, n)), *this);
										if(*ncI != static_cast<size_t>(-1))
											intersectionVertices.insert(nbI());
									}
						}
					}
				}

	}
	return intersectionVertices;
}

//return indecis of neighbouring vertices
std::vector<Eigen::Vector3li> MarchingCubes::getNeighbourCells(const Eigen::Vector3d &position, float radius, bool existing) const
{
	int xDirPositions = static_cast<size_t>(radius / mResolution(0)) + 1;
	int yDirPositions = static_cast<size_t>(radius / mResolution(1)) + 1;
	int zDirPositions = static_cast<size_t>(radius / mResolution(2)) + 1;
	
	std::vector<Eigen::Vector3li> neighbours;
	Eigen::Vector3li baseCell = cell(position);
	Eigen::Vector3li neighbourCell;
	for(int i = -xDirPositions; i <= xDirPositions+1; i++)
	{
		for(int j = -yDirPositions; j <= yDirPositions+1; j++)
		{
			for(int k = -zDirPositions; k <= zDirPositions+1; k++)
			{
				
				if(((i*i + j*j + k*k) * mResolution(0) * mResolution(0)) > radius * radius)
					continue;
				neighbourCell = Vector3li(baseCell(0) + i,baseCell(1) + j, baseCell(2) +  k);
				const CellIndex ncI(cellIndex(neighbourCell), *this);
				if(existing && *ncI == InvPrt)
					continue;
				neighbours.push_back(neighbourCell);
			}
		}
	}

	return neighbours;
}

std::vector<std::pair<size_t, std::array<std::array<int, 3>, 5>>> MarchingCubes::computeIntersectionCells() const
{
	std::vector<std::pair<size_t, std::array<std::array<int, 3>, 5>>> intersectionCells;
	intersectionCells.reserve(mCellToDataIndex.size());
	for(const auto& c : mCellToDataIndex)
	{
		Vector3li cellInd = cell(c.first);
		size_t i = cellInd(0);
		size_t j = cellInd(1);
		size_t k = cellInd(2);

		std::array<bool, 8> ptsConfig;

		for(int l = 0; l < 8; l++)
		{
			float sdf;
			if(!getSDFvalue(i + CELL_VERTICES[l][0],
							j + CELL_VERTICES[l][1],
							k + CELL_VERTICES[l][2], sdf))
			{
				bool res = getSDFvalue(i, j, k, sdf);
				assert(res);
			}
			ptsConfig[l] = sdf < 0;
		}

		std::array<std::array<int, 3>, 5> triangle_type = getMarchingCubesCellTriangulation(ptsConfig);
		if(triangle_type[0][0] == -1)
			continue;
		intersectionCells.push_back(std::make_pair(c.first, triangle_type));
	}
	return intersectionCells;
}



void MarchingCubes::updateSurfaceParticles()
{
	mSurfaceParticlesCount = 0;
	auto& particles = mFluid->getPositions();
	auto& densities = mFluid->getDensities();
	//neighbourhood search
	//TODO: get neighbourhood data from the simulation files
	NeighborhoodSearch ns(mFluid->getCompactSupport());
	ns.add_point_set((Real*)(particles.data()), particles.size(), false);
	mFluid->findNeighbors(ns);
	const auto& neighbors = mFluid->getNeighbors();
	mCurvature = learnSPH::compute_curvature(mFluid.get());
#ifdef DBG
	saveParticlesToVTK("/tmp/" + mSimName + "_ParticleCurvature_" + mFrameNumber + ".vtk", particles, mCurvature);
#endif
	auto samplePoints = [](float radius)
	{
		const size_t numSamples = 100;
		std::vector<Eigen::Vector3d> samples;
		samples.reserve(numSamples);
		for(size_t i = 0; i < numSamples; i++)
		{
			Eigen::Vector3d sample(radius, 0, 0);
			Eigen::AngleAxisd transform(Eigen::Quaterniond::UnitRandom());
			sample = transform * sample;
			samples.push_back(sample);
		}
		return samples;
	};

#ifdef DBG
	saveParticlesToVTK("/tmp/"  + string("RandomSphereSampling") + ".vtk", samplePoints(1));
#endif

	if(mColorFieldSurfaceFactor < 0.05)
	{
		mSurfaceParticlesCount = mFluid->size();
		return;
	}
	
	
	vector<Real> colorField(particles.size(), 0);
	Real r1 = mFluid->getSmoothingLength();
	Real r2 = r1 / 2;
	const auto samples = samplePoints(r1 * 1.1);
	#pragma omp parallel for schedule(static)
	for(size_t i = 0; i < particles.size(); i++)
	{
		auto leftSamples = samples;
		for(size_t j : neighbors[i][0])
		{
			for(int currentSample = leftSamples.size() - 1; currentSample >= 0; currentSample--)
			{
				if((particles[i] + samples[currentSample] - particles[j]).squaredNorm() < r2 * r2)
				{
					//remove samples from sample
					auto tmp = leftSamples.back();
					leftSamples.back() = leftSamples[currentSample];
					leftSamples[currentSample] = tmp;
					leftSamples.pop_back();
				}
			}
		}
		colorField[i] = static_cast<Real>(leftSamples.size()) / samples.size();
	}
	//smooth out color field
	auto smoothedColorField = colorField;
	for(size_t k = 0; k < 2; k++)
	{
		#pragma omp parallel for schedule(static)
		for(size_t i = 0; i < particles.size(); i++)
		{
			Real totalWeights = learnSPH::kernel::kernelFunction(particles[i], particles[i], r1);
			smoothedColorField[i] = totalWeights * colorField[i];
			for(size_t j : neighbors[i][0])
			{
				auto weight = learnSPH::kernel::kernelFunction(particles[i], particles[j], r1);
				smoothedColorField[i] += weight * colorField[j];
				totalWeights += weight;
			}
			smoothedColorField[i] /= totalWeights;
		}
		colorField.swap(smoothedColorField);
	}
#ifdef DBG
	saveParticlesToVTK("/tmp/" + mSimName +
						"MarchingCubesColorField_" + mFrameNumber + ".vtk",
					   particles, colorField);

#endif
	
	//relocate all surface particles at the beginning of the array
	for(size_t i = 0; i < particles.size(); i++)
	{
		if(colorField[i] > mColorFieldSurfaceFactor)
		{
			particles[i].swap(particles[mSurfaceParticlesCount]);
			auto tmp = densities[i];
			densities[i] = densities[mSurfaceParticlesCount];
			densities[mSurfaceParticlesCount] = tmp;
			tmp = mCurvature[i];
			mCurvature[i] = mCurvature[mSurfaceParticlesCount];
			mCurvature[mSurfaceParticlesCount] = tmp;
			mSurfaceParticlesCount++;
		} 
	}
	mCurvature.resize(mSurfaceParticlesCount);
#ifdef DBG
	std::vector<Eigen::Vector3d> surfaceParticles(particles.begin(), particles.begin() + mSurfaceParticlesCount);
	saveParticlesToVTK("/tmp/" + mSimName +
						"_MarchingCubesSPHSurfacePrticles_" + mFrameNumber + ".vtk",
					   surfaceParticles);

#endif
}
