#pragma once
#include <memory>
#include <Eigen/Dense>
#include "SurfaceReconstructor.hpp"
#include <learnSPH/core/storage.h>


class MarchingCubes : public SurfaceReconstructor
{
protected:
	std::shared_ptr<learnSPH::FluidSystem> mFluid;
	Eigen::Vector3d mLowerCorner;
	Eigen::Vector3d mUpperCorner;
	Eigen::Vector3i mDimentions;
	Eigen::Vector3d mResolution;
	float mInitialValue {-0.5};

public:
    MarchingCubes() = delete;
    MarchingCubes(const MarchingCubes& other):
		mLowerCorner(other.mLowerCorner),
		mUpperCorner(other.mUpperCorner),
		mDimentions(other.mDimentions),
		mResolution(other.mResolution),
        mInitialValue(other.mInitialValue)
	{
	}
    MarchingCubes& operator=(const MarchingCubes&) = delete;
	
    explicit MarchingCubes(std::shared_ptr<learnSPH::FluidSystem> fluid,
								const Eigen::Vector3d lCorner,
								const Eigen::Vector3d uCorner,
								const Eigen::Vector3d cResolution,
								float initValue);
	
protected:
    void setFluidSystem(std::shared_ptr<learnSPH::FluidSystem> fluid) { mFluid = fluid; }
    virtual void updateGrid() = 0;
    virtual void updateLevelSet() = 0;
    std::vector<Eigen::Vector3d> getTriangles() const;
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
    virtual float getSDFvalue(int i, int j, int k) const = 0;
};

class NaiveMarchingCubes : public MarchingCubes
{
public:
    explicit NaiveMarchingCubes(std::shared_ptr<learnSPH::FluidSystem> fluid,
                           const Eigen::Vector3d lCorner,
                           const Eigen::Vector3d uCorner,
                           const Eigen::Vector3d cResolution,
                           float initValue ):
        MarchingCubes(fluid, lCorner, uCorner, cResolution, initValue),
        mLevelSetFunction(mDimentions(0) * mDimentions(1) * mDimentions(2))
    {}
    explicit NaiveMarchingCubes(const NaiveMarchingCubes& other):
        MarchingCubes(other),
        mLevelSetFunction(other.mLevelSetFunction)
    {}
    std::vector<Eigen::Vector3d> generateMesh(const std::shared_ptr<learnSPH::FluidSystem> fluid) override;

private:
    void updateGrid() override;
    void updateLevelSet() override;
    float getSDFvalue(int i, int j, int k) const override
    {
        return mLevelSetFunction[cellIndex(Eigen::Vector3i(i, j, k))];
    }


    std::vector<float> mLevelSetFunction;
};
