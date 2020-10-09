#include <memory>
#include <Eigen/Dense>
#include <learnSPH/core/storage.h>
#include <learnSPH/core/kernel.h>
#include <learnSPH/core/vtk_writer.h>
#include "look_up_tables.hpp"
#include "NaiveMarchingCubes.hpp"
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
}

void NaiveMarchingCubes::configureHashTables()
{
	MarchingCubes::configureHashTables();
	mLevelSetFunction.max_load_factor(mSurfaceCells.max_load_factor());
}

std::vector<Eigen::Vector3d> MarchingCubes::generateMesh(const std::shared_ptr<learnSPH::FluidSystem> fluid)
{
	setFluidSystem(fluid);
	updateSurfaceParticles();
	configureHashTables();
	updateGrid();
	updateLevelSet();
	
#ifdef DEBUG
	static int cnt = 0;
	vector<Real> sdf;
	vector<Vector3R> vertices;
	for(const auto& vert : mSurfaceCells)
	{
		auto cI = cell(vert.first);
		auto cC = cellCoord(cI);
		auto sdfV = getSDFvalue(cI(0), cI(1), cI(2));
		vertices.push_back(cC);
		sdf.push_back(sdfV);
	}
	saveParticlesToVTK("/tmp/SDF" + to_string(cnt) + ".vtk", vertices, sdf);
	cnt++;
#endif
	auto mesh = getTriangles();
	return mesh;
}
	
	
void NaiveMarchingCubes::updateGrid()
{
	mSurfaceCells.clear();
	mLevelSetFunction.clear();
	mPartPerSupportArea = (mFluid->getCompactSupport() * mFluid->getCompactSupport() * mFluid->getCompactSupport()) / 
							(mFluid->getDiameter() * mFluid->getDiameter() * mFluid->getDiameter());
	const auto& particles = mFluid->getPositions();
	for(int i = 0; i < mSurfaceParticlesCount; i++)
	{
		auto nCells = getNeighbourCells(particles[i], mFluid->getCompactSupport()/2, false);
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
void NaiveMarchingCubes::updateLevelSet()
{
	if(!mFluid)
		throw std::runtime_error("fluid was not initialised");
	
	const vector<Vector3R>& positions = mFluid->getPositions();
	const vector<Real>& densities = mFluid->getDensities();
	
	for(int i = 0; i < mFluid->size(); i++)
	{
		double fluidDensity = densities[i];
		const Eigen::Vector3d& particle = positions[i];
		std::vector<Eigen::Vector3i> neighbourCells = 
				std::move(getNeighbourCells(particle, mFluid->getCompactSupport()));
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


	for(const auto& cellVert : mSurfaceCells)
	{
		Vector3i cellInd = cell(cellVert.first);
		size_t i = cellInd(0);
		size_t j = cellInd(1);
		size_t k = cellInd(2);
		
		std::array<bool, 8> ptsConfig;

		for(int l = 0; l < 8; l++) 
		{
			try {
				ptsConfig[l] = getSDFvalue(
							i + CELL_VERTICES[l][0], 
							j + CELL_VERTICES[l][1], 
							k + CELL_VERTICES[l][2]) < 0;
			} catch (...) {
				ptsConfig[l] = getSDFvalue(i, j, k) < 0;
			}
		}

		std::array<std::array<int, 3>, 5> triangle_type = getMarchingCubesCellTriangulation(ptsConfig);

		for(size_t l = 0; l < 5; l++) {
			
			for(size_t m = 0; m < 3; m++) {
				if(triangle_type[l][m] == -1) break;
				
				int c1Xind = (i + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][0]);
				int c1Yind = (j + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][1]);
				int c1Zind = (k + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][2]);

				int c2Xind = (i + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][0]);
				int c2Yind = (j + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][1]);
				int c2Zind = (k + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][2]);
				float v1, v2;
				
				try { v1 = getSDFvalue(c1Xind, c1Yind, c1Zind); }
				catch (...) { v1 = getSDFvalue(i,j,k); }
				
				try { v2 = getSDFvalue(c2Xind, c2Yind, c2Zind); }
				catch (...) { v2 = getSDFvalue(i,j,k); }

				Eigen::Vector3d p1 = cellCoord(Eigen::Vector3i(c1Xind, c1Yind, c1Zind));
				Eigen::Vector3d p2 = cellCoord(Eigen::Vector3i(c2Xind, c2Yind, c2Zind));
				triangleMesh.push_back(lerp(p1, p2, v1, v2, 0));
			}
		}
	}
	return triangleMesh;
}



//return indecis of neighbouring vertices
std::vector<Eigen::Vector3i> MarchingCubes::getNeighbourCells(const Eigen::Vector3d &position, float radius, bool existing) const
{
	int xDirPositions = static_cast<size_t>(radius / mResolution(0)) + 1;
	int yDirPositions = static_cast<size_t>(radius / mResolution(1)) + 1;
	int zDirPositions = static_cast<size_t>(radius / mResolution(2)) + 1;
	
	std::vector<Eigen::Vector3i> neighbours;
	Eigen::Vector3i baseCell = cell(position);
	Eigen::Vector3i neighbourCell;
	for(int i = -xDirPositions; i <= xDirPositions+1; i++)
	{
		for(int j = -yDirPositions; j <= yDirPositions+1; j++)
		{
			for(int k = -zDirPositions; k <= zDirPositions+1; k++)
			{
				neighbourCell = baseCell + Vector3i(i, j, k);
				
				if(existing && mSurfaceCells.find(cellIndex(neighbourCell)) == mSurfaceCells.end())
					continue;
				

				Eigen::Vector3d neighbour = cellCoord(neighbourCell);
				
				if((position - neighbour).squaredNorm() < radius * radius)
					neighbours.push_back(neighbourCell);
			}
		}
	}

	return neighbours;
}


void MarchingCubes::updateSurfaceParticles()
{
#if 0
	mSurfaceParticlesCount = 0;
	auto& particles = mFluid->getPositions();
	auto& densities = mFluid->getDensities();
	
	//neighbourhood search
	//TODO: get neighbourhood data from the simulation files
	NeighborhoodSearch ns(mFluid->getCompactSupport());
	ns.add_point_set((Real*)(particles.data()), particles.size(), false);
	mFluid->findNeighbors(ns);
	const auto& neighbors = mFluid->getNeighbors();
	
	vector<Real> colorField;
	#pragma omp parallel for schedule(static)
	for(int i = 0; i < particles.size(); i++)
	{
		Real cf = 0;
		for(size_t j : neighbors[i][0])
			cf += (learnSPH::kernel::kernelFunction(particles[i], particles[j], mFluid->getSmoothingLength()) / densities[j]);
		cf *= mFluid->getMass();
		colorField.push_back(cf);
	}
	
	//relocate all surface particles at the beginning of the array
	for(int i = 0; i < particles.size(); i++)
	{
		if(colorField[i] < mColorFieldSurfaceFactor)
		{
			particles[i].swap(particles[mSurfaceParticlesCount]);
			auto tmp = densities[i];
			densities[i] = densities[mSurfaceParticlesCount];
			densities[mSurfaceParticlesCount] = tmp;
			mSurfaceParticlesCount++;
		} 
	}
#else
	mSurfaceParticlesCount = mFluid->size();
#endif
}
