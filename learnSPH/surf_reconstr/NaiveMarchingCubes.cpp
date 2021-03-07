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
		mSurfaceCells.max_load_factor(2);
	mSurfaceCells.rehash(mFluid->size() * 2);
}

void NaiveMarchingCubes::configureHashTables()
{
	MarchingCubes::configureHashTables();
	mLevelSetFunction.max_load_factor(mSurfaceCells.max_load_factor());
	mLevelSetFunction.rehash(mSurfaceCells.bucket_count());
}

std::vector<Eigen::Vector3d> MarchingCubes::generateMesh(const std::shared_ptr<learnSPH::FluidSystem> fluid)
{
	setFluidSystem(fluid);
	globalPerfStats.startTimer("updateSurfaceParticles");
	updateSurfaceParticles();
	globalPerfStats.stopTimer("updateSurfaceParticles");

	globalPerfStats.startTimer("configureHashTables");
	configureHashTables();
	globalPerfStats.stopTimer("configureHashTables");

	globalPerfStats.startTimer("updateGrid");
	updateGrid();
	globalPerfStats.stopTimer("updateGrid");

	globalPerfStats.startTimer("updateLevelSet");
	updateLevelSet();
	globalPerfStats.stopTimer("updateLevelSet");
#ifdef DBG
	vector<Real> sdf;
	vector<Vector3R> vertices;
	vector<Real> cellCurvature;
	vector<Real> particleConcentration;
	for(const auto& vert : mSurfaceCells)
	{
		auto cI = cell(vert.first);
		auto cC = cellCoord(cI);
		float sdfV; 
		bool res = getSDFvalue(cI(0), cI(1), cI(2), sdfV);
		assert(res);
		vertices.push_back(cC);
		sdf.push_back(sdfV);
		Real curvature; getCurvature(cellIndex(cI), curvature);
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
		intersectionCellVerticePoints.push_back(cellCoord(cell(c.first)));
		Real curvature; getCurvature(c.first, curvature);
		intersectionCellCurvature.push_back(curvature);
	}
	saveParticlesToVTK("/tmp/" + mSimName + "IntersectionCellsCurvature_" + mFrameNumber + ".vtk", intersectionCellVerticePoints, intersectionCellCurvature);

#endif
	globalPerfStats.startTimer("getTriangles");
	auto mesh = getTriangles();
	globalPerfStats.stopTimer("getTriangles");
	return mesh;
}
	
	
void NaiveMarchingCubes::updateGrid()
{
	mSurfaceCells.clear();
	mLevelSetFunction.clear();
	mPartPerSupportArea = 8 * (mFluid->getSmoothingLength() * mFluid->getSmoothingLength() * mFluid->getSmoothingLength()) /
							(mFluid->getDiameter() * mFluid->getDiameter() * mFluid->getDiameter());
	const auto& particles = mFluid->getPositions();
	for(size_t i = 0; i < mSurfaceParticlesCount; i++)
	{
		auto nCells = getNeighbourCells(particles[i], mFluid->getCompactSupport(), false);
		for(const auto& nc : nCells)
		{
			auto cI = cellIndex(nc);
			if(mSurfaceCells.find(cI) == mSurfaceCells.end())
			{
				mSurfaceCells[cI] = 1;
				mSurfaceCellsCurvature[cI] = mCurvature[i];
			}
			else
			{
				mSurfaceCells[cI]++;
				mSurfaceCellsCurvature[cI] += mCurvature[i];
			}
		}
	}
	//no need in particle curvature any more. Free space
	mCurvature.clear();
}

void NaiveMarchingCubes::updateLevelSet()
{
	if(!mFluid)
		throw std::runtime_error("fluid was not initialised");
	
	const vector<Vector3R>& positions = mFluid->getPositions();
	const vector<Real>& densities = mFluid->getDensities();
	
	for(size_t i = 0; i < mFluid->size(); i++)
	{
		double fluidDensity = densities[i];
		const Eigen::Vector3d& particle = positions[i];
		std::vector<Eigen::Vector3li> neighbourCells =
				getNeighbourCells(particle, mFluid->getCompactSupport());
		for(const auto& cell : neighbourCells)
		{
			float weight = learnSPH::kernel::kernelFunction(
				particle, 
				cellCoord(cell), 
				mFluid->getSmoothingLength()
			);
			float density = std::max(fluidDensity, mFluid->getRestDensity());
			float total = mFluid->getMass() / density * weight;
			if(mLevelSetFunction.find(cellIndex(cell)) == mLevelSetFunction.end())
				mLevelSetFunction[cellIndex(cell)] = mInitialValue;
			mLevelSetFunction[cellIndex(cell)] += total;
		}
	}
}

vector<Eigen::Vector3d> MarchingCubes::getTriangles() const
{

	vector<Eigen::Vector3d> triangleMesh;
	triangleMesh.reserve(mSurfaceCells.size() * 3 * 3);

	auto intersectionCells = computeIntersectionCells();

#pragma omp parallel for schedule(static)
	for(size_t g = 0; g < intersectionCells.size(); g++)
	{
		const auto& cellVert = intersectionCells[g];
		Eigen::Vector3li cellInd = cell(cellVert.first);
		size_t i = cellInd(0);
		size_t j = cellInd(1);
		size_t k = cellInd(2);
		
		const std::array<std::array<int, 3>, 5>& triangle_type = cellVert.second;

		for(size_t l = 0; l < 5; l++) {
			std::array<Eigen::Vector3d, 3> triangle;
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
				triangle[m] = lerp(p1, p2, v1, v2, 0);
			}
#pragma omp critical(UpdateTriangleMesh)
{
			triangleMesh.insert(triangleMesh.end(), triangle.begin(), triangle.end());
}
		}
	}
	return triangleMesh;
}

std::unordered_map<size_t, size_t> MarchingCubes::computeIntersectionCellVertices(int neighborsCnt) const
{
	std::unordered_map<size_t, size_t> intersectionCellVertices;
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

						size_t ncI = cellIndex(nc + Eigen::Vector3li(i, j, k));
						if(intersectionCellVertices.find(ncI) != intersectionCellVertices.end())
							continue;
						auto fluidCell = mSurfaceCells.find(ncI);
						if(fluidCell == mSurfaceCells.end())
							continue;
						intersectionCellVertices.insert(*fluidCell);
					}
				}
			}
		}
	}
	return intersectionCellVertices;
}

std::unordered_map<size_t, size_t> MarchingCubes::computeIntersectionVertices(int neighbors) const
{
	auto allVertices = mSurfaceCells;
	std::unordered_map<size_t, size_t> intersectionVertices;
	for(auto vert : mSurfaceCells)
	{
		float sdf; bool res = getSDFvalue(vert.first, sdf);
		assert(res);
		auto c = cell(vert.first);
		for(int i = -1; i <= 1; i++)
			for(int j = -1; j <= 1; j++)
				for(int k = -1; k <= 1; k++)
				{
					size_t ncI = cellIndex(c + Eigen::Vector3li(i,j,k));
					float nbSdf; res = getSDFvalue(ncI, nbSdf);
					if(!res)
						continue;
					if(nbSdf * sdf < 0)
					{
						intersectionVertices[vert.first] = vert.second;
						intersectionVertices[ncI] = mSurfaceCells.at(ncI);
						if(neighbors > 0)
						{
							for(int l = -neighbors; l <= neighbors; l++)
								for(int m = -neighbors; m <= neighbors; m++)
									for(int n = -neighbors; n <= neighbors; n++)
									{
										auto nbI = cellIndex(c + Eigen::Vector3li(l, m, n));
										auto element = mSurfaceCells.find(nbI);
										if(element != mSurfaceCells.end())
											intersectionVertices.insert(*element);
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
				if(existing && mSurfaceCells.find(cellIndex(neighbourCell)) == mSurfaceCells.end())
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
	intersectionCells.reserve(mSurfaceCells.size());
	for(const auto& c : mSurfaceCells)
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
	saveParticlesToVTK("/tmp/" + mSimName + "ParticleCurvature" + mFrameNumber + ".vtk", particles, mCurvature);
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
	const auto samples = samplePoints(r1);
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
						"MarchingCubesSPHSurfacePrticles_" + mFrameNumber + ".vtk",
					   surfaceParticles);

#endif
}
