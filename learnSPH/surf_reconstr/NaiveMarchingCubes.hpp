#pragma once
#include <memory>
#include <Eigen/Dense>
#include "SurfaceReconstructor.hpp"
#include <learnSPH/core/storage.h>


class NaiveMarchingCubes : public SurfaceReconstructor
{
	std::shared_ptr<learnSPH::FluidSystem> mFluid;
	Eigen::Vector3d mLowerCorner;
	Eigen::Vector3d mUpperCorner;
	Eigen::Vector3i mDimentions;
	Eigen::Vector3d mResolution;
	std::vector<float> mLevelSetFunction;

public:
	NaiveMarchingCubes() = delete;
	NaiveMarchingCubes(const NaiveMarchingCubes&) = delete;
	NaiveMarchingCubes& operator=(const NaiveMarchingCubes&) = delete;
	
	explicit NaiveMarchingCubes(std::shared_ptr<learnSPH::FluidSystem> fluid,
								const Eigen::Vector3d lCorner,
								const Eigen::Vector3d uCorner,
								const Eigen::Vector3d cResolution,
								float initValue);
	
	const std::vector<float>& levelSet() const { return mLevelSetFunction; }
	
	void updateGrid() override;
	void updateLevelSet() override;
	vector<Eigen::Vector3d> getTriangles() const override;
private:
	///linear interpolation between two vectors given 2 float values and target value
	Eigen::Vector3d lerp(const Eigen::Vector3d& a,
						 const Eigen::Vector3d& b, 
						 float av, 
						 float bv, 
						 float tv) const;
	
	///get nearest cell indeces given coordinate in space
	Eigen::Vector3i cell(const Eigen::Vector3d& vec) const;
	
	///Calculate cell vertice coordinate given cell indecis
	Eigen::Vector3d cellCoord(const Eigen::Vector3i& vec) const;
	
	///get array index from cell indeces
	int cellIndex(const Eigen::Vector3i& ind) const;
	
	///Calculate cell indeces of neighbour particles
	std::vector<Eigen::Vector3i> getNeighbourCells(const Eigen::Vector3d& position, float radius) const;
	

	
};
