#include <memory>
#include <Eigen/Dense>
#include <learnSPH/core/storage.h>
#include <learnSPH/core/kernel.h>
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
{}

std::vector<Eigen::Vector3d> NaiveMarchingCubes::generateMesh(const std::shared_ptr<learnSPH::FluidSystem> fluid)
{
    setFluidSystem(fluid);
    updateGrid();
    updateLevelSet();
    auto mesh = std::move(getTriangles());
    return mesh;
}
	
	
void NaiveMarchingCubes::updateGrid()
{
	mLevelSetFunction.assign(this->mLevelSetFunction.size(), mInitialValue);
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
			mLevelSetFunction[cellIndex(cell)] += total;
		}
	}
}
	
vector<Eigen::Vector3d> MarchingCubes::getTriangles() const
{

	vector<Eigen::Vector3d> triangleMesh;
	triangleMesh.reserve(mDimentions(0) * mDimentions(1) * mDimentions(2) * 12);

	for(size_t i = 0; i < mDimentions(0) - 1; i++) {

		for(size_t j = 0; j < mDimentions(1) - 1; j++) {

			for(size_t k = 0; k < mDimentions(2) - 1; k++) {

				std::array<bool, 8> ptsConfig;

				for(int l = 0; l < 8; l++) 
                    ptsConfig[l] = getSDFvalue(
								i + CELL_VERTICES[l][0], 
								j + CELL_VERTICES[l][1], 
                                k + CELL_VERTICES[l][2]) < 0;

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
						
                        float v1 = getSDFvalue(c1Xind, c1Yind, c1Zind);
                        float v2 = getSDFvalue(c2Xind, c2Yind, c2Zind);
						float factor = v1 / (v1 - v2);
						Eigen::Vector3d p1 = cellCoord(Eigen::Vector3i(c1Xind, c1Yind, c1Zind));
						Eigen::Vector3d p2 = cellCoord(Eigen::Vector3i(c2Xind, c2Yind, c2Zind));

						triangleMesh.push_back(lerp(p1, p2, v1, v2, 0));
					}
				}
			}
		}
	}
	return triangleMesh;
}

Eigen::Vector3d MarchingCubes::lerp(const Eigen::Vector3d& a,const Eigen::Vector3d& b, float av, float bv, float tv) const
{
	float factor = (tv - av)/(bv - av);
	return a * (1 - factor) + factor * b;
}

Eigen::Vector3i MarchingCubes::cell(const Eigen::Vector3d& vec) const
{
	float xf = (vec(0) - mLowerCorner(0)) / (mUpperCorner(0) - mLowerCorner(0));
	float yf = (vec(1) - mLowerCorner(1)) / (mUpperCorner(1) - mLowerCorner(1));
	float zf = (vec(2) - mLowerCorner(2)) / (mUpperCorner(2) - mLowerCorner(2));
	return Eigen::Vector3i(std::floor(mDimentions(0) * xf),
						   std::floor(mDimentions(1) * yf),
						   std::floor(mDimentions(2) * zf));

}
Eigen::Vector3d MarchingCubes::cellCoord(const Eigen::Vector3i& vec) const
{
	float xf = static_cast<double>(vec(0)) / mDimentions(0);
	float yf = static_cast<double>(vec(1)) / mDimentions(1);
	float zf = static_cast<double>(vec(2)) / mDimentions(2);
	
	double x = mLowerCorner(0) * (1- xf) + mUpperCorner(0) * xf;
	double y = mLowerCorner(1) * (1- yf) + mUpperCorner(1) * yf;
	double z = mLowerCorner(2) * (1- zf) + mUpperCorner(2) * zf;
	return Eigen::Vector3d(x, y, z);
	
}
int MarchingCubes::cellIndex(const Eigen::Vector3i& ind) const
{
	return ind(0) * mDimentions(1) * mDimentions(2) + 
			ind(1) * mDimentions(2) + 
			ind(2);
}


//return indecis of neighbouring vertices
std::vector<Eigen::Vector3i> MarchingCubes::getNeighbourCells(const Eigen::Vector3d& position, float radius) const
{
	int xDirPositions = static_cast<size_t>(radius / mResolution(0)) + 1;
	int yDirPositions = static_cast<size_t>(radius / mResolution(1)) + 1;
	int zDirPositions = static_cast<size_t>(radius / mResolution(2)) + 1;
	
	std::vector<Eigen::Vector3i> neighbours;
	Eigen::Vector3i baseCell = cell(position);
	Eigen::Vector3i neighbourCell;
	for(int i = -xDirPositions; i <= xDirPositions; i++)
	{
		neighbourCell(0) = baseCell(0) + i;
		for(int j = -yDirPositions; j <= yDirPositions; j++)
		{
			neighbourCell(1) = baseCell(1) + j;
			for(int k = -zDirPositions; k <= zDirPositions; k++)
			{
				neighbourCell(2) = baseCell(2) + k;
				
				//ignore neighbours that are out of domain
				Eigen::Vector3i check = neighbourCell;
				if(check(0) < 0 || check(1) < 0 || check(2) < 0)
					continue;
				check = mDimentions - Eigen::Vector3i(1, 1, 1) - neighbourCell;
				if(check(0) < 0 || check(1) < 0 || check(2) < 0)
					continue;
				

				assert(cellIndex(neighbourCell) < mLevelSetFunction.size() && cellIndex(neighbourCell) >= 0);
				Eigen::Vector3d neighbour = cellCoord(neighbourCell);
				
				if((position - neighbour).squaredNorm() < radius * radius)
					neighbours.push_back(neighbourCell);
			}
		}
	}

	return neighbours;
}


	
