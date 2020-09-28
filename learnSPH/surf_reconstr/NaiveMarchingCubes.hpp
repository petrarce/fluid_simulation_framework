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
	size_t mSurfaceParticlesCount {0};
	Real mColorFieldSurfaceFactor {0.8};
	float mInitialValue {-0.5};
	std::unordered_map<size_t, size_t> mSurfaceCells;

public:
	MarchingCubes() = delete;
	explicit MarchingCubes(const MarchingCubes& other):
		mLowerCorner(other.mLowerCorner),
		mUpperCorner(other.mUpperCorner),
		mDimentions(other.mDimentions),
		mResolution(other.mResolution),
		mColorFieldSurfaceFactor(other.mColorFieldSurfaceFactor),
		mInitialValue(other.mInitialValue)
	{
	}
	MarchingCubes& operator=(const MarchingCubes&) = delete;
	
	explicit MarchingCubes(std::shared_ptr<learnSPH::FluidSystem> fluid,
		const Eigen::Vector3d lCorner,
		const Eigen::Vector3d uCorner,
		const Eigen::Vector3d cResolution,
		float initValue);
	std::vector<Eigen::Vector3d> generateMesh(const std::shared_ptr<learnSPH::FluidSystem> fluid) override;
	void setColorFieldFactor(Real factor) { mColorFieldSurfaceFactor = factor; }
	
protected:
	void setFluidSystem(std::shared_ptr<learnSPH::FluidSystem> fluid) { mFluid = fluid; }
	virtual void updateGrid() = 0;
	virtual void updateLevelSet() = 0;
	virtual void configureHashTables();
	virtual void updateSurfaceParticles();
	virtual float getSDFvalue(int i, int j, int k) const = 0;
	inline float getSDFvalue(const Vector3i& c) const
	{
		return getSDFvalue(c(0), c(1), c(2));
	}
	std::vector<Eigen::Vector3d> getTriangles() const;
	///Calculate cell indeces of neighbour cells
	std::vector<Eigen::Vector3i> getNeighbourCells(const Eigen::Vector3d& position, float radius, bool existing = true) const;

	///linear interpolation between two vectors given 2 float values and target value
	inline Eigen::Vector3d lerp(const Eigen::Vector3d& a,
		const Eigen::Vector3d& b, 
		float av, 
		float bv, 
		float tv) const
	{
		float factor = (tv - av)/(bv - av);
		return a * (1 - factor) + factor * b;
	}
	
	
	///get nearest cell indeces given coordinate in space
	inline Eigen::Vector3i cell(const Eigen::Vector3d& vec) const
	{
		float xf = (vec(0) - mLowerCorner(0)) / (mUpperCorner(0) - mLowerCorner(0));
		float yf = (vec(1) - mLowerCorner(1)) / (mUpperCorner(1) - mLowerCorner(1));
		float zf = (vec(2) - mLowerCorner(2)) / (mUpperCorner(2) - mLowerCorner(2));
		return Eigen::Vector3i(std::floor(mDimentions(0) * xf),
							   std::floor(mDimentions(1) * yf),
							   std::floor(mDimentions(2) * zf));
	
	}
	
	inline Eigen::Vector3i cell(size_t index) const
	{
		int k = index % mDimentions(2);
		int j = (index / mDimentions(2)) % mDimentions(1);
		int i = ((index / mDimentions(2)) / mDimentions(1));
		return Vector3i(i, j, k);
	}
	
	///Calculate cell vertice coordinate given cell indecis
	inline Eigen::Vector3d cellCoord(const Eigen::Vector3i& vec) const
	{
		float xf = static_cast<double>(vec(0)) / mDimentions(0);
		float yf = static_cast<double>(vec(1)) / mDimentions(1);
		float zf = static_cast<double>(vec(2)) / mDimentions(2);
		
		double x = mLowerCorner(0) * (1- xf) + mUpperCorner(0) * xf;
		double y = mLowerCorner(1) * (1- yf) + mUpperCorner(1) * yf;
		double z = mLowerCorner(2) * (1- zf) + mUpperCorner(2) * zf;
		return Eigen::Vector3d(x, y, z);
		
	}
	
	///get array index from cell indeces
	inline int cellIndex(const Eigen::Vector3i& ind) const
	{
		return ind(0) * mDimentions(1) * mDimentions(2) + 
				ind(1) * mDimentions(2) + 
				ind(2);
	}
	
	

};

class NaiveMarchingCubes : public MarchingCubes
{
public:
	explicit NaiveMarchingCubes(std::shared_ptr<learnSPH::FluidSystem> fluid,
		const Eigen::Vector3d lCorner,
		const Eigen::Vector3d uCorner,
		const Eigen::Vector3d cResolution,
		float initValue
	):
		MarchingCubes(fluid, lCorner, uCorner, cResolution, initValue)
	{
	}
	explicit NaiveMarchingCubes(const NaiveMarchingCubes& other):
		MarchingCubes(other),
		mLevelSetFunction(other.mLevelSetFunction)
	{}

protected:
	void updateGrid() override;
	void updateLevelSet() override;
	void configureHashTables() override;
	
	float getSDFvalue(int i, int j, int k) const override
	{
		auto val = mLevelSetFunction.find(cellIndex(Eigen::Vector3i(i, j, k)));
		if(val  == mLevelSetFunction.end())
			return mInitialValue;
		return val->second;
	}


	std::unordered_map<int, float> mLevelSetFunction;
};
