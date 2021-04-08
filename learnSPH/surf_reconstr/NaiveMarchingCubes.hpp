#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

#include <Eigen/Dense>

#include "SurfaceReconstructor.hpp"
#include <learnSPH/core/storage.h>

const size_t InvPrt = static_cast<size_t>(-1);

namespace Eigen
{
typedef Matrix<int64_t, 3, 1> Vector3li;
}
class MarchingCubes : public SurfaceReconstructor
{
protected:

	std::shared_ptr<learnSPH::FluidSystem> mFluid;
	Eigen::Vector3d mLowerCorner;
	Eigen::Vector3d mUpperCorner;
	Eigen::Vector3li mDimentions;
	Eigen::Vector3d mResolution;
	size_t mSurfaceParticlesCount {0};
	Real mColorFieldSurfaceFactor {0.8};
	float mInitialValue {-0.5};
	std::unordered_map<size_t/*index*/, size_t/*number of particles in the neighbourhood*/> mDataToCellIndex;
	std::unordered_map<size_t/*index*/, size_t/*number of particles in the neighbourhood*/> mCellToDataIndex;
	std::vector<Real> mMcVertexCurvature;
	std::vector<Real> mMcVertexSdf;
	std::vector<Real> mCurvature;
	std::vector<size_t> mMcVertexSphParticles;
	int mPartPerSupportArea {0};
	string mFrameNumber {"UNDEFINED"};
	std::string mSimName;


public:
	MarchingCubes() = delete;
	explicit MarchingCubes(const MarchingCubes& other):
		mLowerCorner(other.mLowerCorner),
		mUpperCorner(other.mUpperCorner),
		mDimentions(other.mDimentions),
		mResolution(other.mResolution),
		mColorFieldSurfaceFactor(other.mColorFieldSurfaceFactor),
		mInitialValue(other.mInitialValue),
		mDataToCellIndex(other.mDataToCellIndex),
		mCellToDataIndex(other.mCellToDataIndex),
		mMcVertexCurvature(other.mMcVertexCurvature),
		mMcVertexSdf(other.mMcVertexSdf),
		mCurvature(other.mCurvature),
		mMcVertexSphParticles(other.mMcVertexSphParticles),
		mPartPerSupportArea(other.mPartPerSupportArea),
		mFrameNumber(other.mFrameNumber),
		mSimName(other.mSimName)

	{
	}
	MarchingCubes& operator=(const MarchingCubes&) = delete;
	
	void setFrameNumber(const string& frame) { mFrameNumber = frame; }
	explicit MarchingCubes(std::shared_ptr<learnSPH::FluidSystem> fluid,
		const Eigen::Vector3d lCorner,
		const Eigen::Vector3d uCorner,
		const Eigen::Vector3d cResolution,
		float initValue);
	std::vector<Eigen::Vector3d> generateMesh(const std::shared_ptr<learnSPH::FluidSystem> fluid) override;
	void setColorFieldFactor(Real factor) { mColorFieldSurfaceFactor = factor; }
	void setSimName(const std::string& name) { mSimName = name; }
	virtual void clearBuffers()
	{
		mMcVertexCurvature.clear(); mMcVertexCurvature.shrink_to_fit();
		mMcVertexSdf.clear(); mMcVertexSdf.shrink_to_fit();
		mMcVertexSphParticles.clear(); mMcVertexSphParticles.shrink_to_fit();
	}
	
protected:
	class CellIndex{
	private:
		size_t mIndex;
		const MarchingCubes& mMarchingCubes;
		mutable bool mDataIndexRequested {false};
		mutable size_t mDataIndex {InvPrt};
		mutable std::mutex mLock;
	public:
		CellIndex(size_t index, const MarchingCubes& mc):
			mIndex(index),
			mMarchingCubes(mc)
		{}
		size_t operator*() const
		{
			if(!mDataIndexRequested)
			{
				std::lock_guard<std::mutex> lock(mLock);
				auto dataIndex = mMarchingCubes.mCellToDataIndex.find(mIndex);
				if(dataIndex != mMarchingCubes.mCellToDataIndex.end())
					mDataIndex = dataIndex->second;
				mDataIndexRequested = true;
			}
			return mDataIndex;
		}
		size_t operator()() const
		{
			return mIndex;
		}
		const CellIndex& operator=(size_t cI)
		{
			if(cI == mIndex)
				return *this;

			std::lock_guard<std::mutex> lock(mLock);
			mIndex = cI;
			mDataIndex = InvPrt;
			mDataIndexRequested = false;
			return *this;
		}
	};
	class DataIndex{
		private:
			size_t mIndex;
			const MarchingCubes& mMarchingCubes;
			mutable size_t mCellIndexRequested {false};
			mutable size_t mCellIndex {InvPrt};
			mutable std::mutex mLock;

		public:
			DataIndex(size_t index, const MarchingCubes& mc):
				mIndex(index),
				mMarchingCubes(mc)
			{}

			size_t operator*() const
			{
				if(!mCellIndexRequested)
				{
					std::lock_guard<std::mutex> lock(mLock);
					auto cellIndex = mMarchingCubes.mDataToCellIndex.find(mIndex);
					if(cellIndex != mMarchingCubes.mDataToCellIndex.end())
						mCellIndex = cellIndex->second;
					mCellIndexRequested = true;
				}
				return mCellIndex;
			}

			size_t operator()() const
			{
				return mIndex;
			}
	};

	friend class CellIndex;
	friend class DataIndex;

	void setFluidSystem(std::shared_ptr<learnSPH::FluidSystem> fluid) { mFluid = fluid; }
	void relocateFluidParticles(Real supportRadius);
	virtual void updateGrid() = 0;
	virtual void updateLevelSet() = 0;
	void configureHashTables();
	bool getSDFvalue(const CellIndex& cI, float& sdf) const
	{
		size_t dataIndex = *cI;
		if(dataIndex == static_cast<size_t>(-1))
			return false;
		assert(dataIndex < mMcVertexSdf.size());
		sdf = mMcVertexSdf[dataIndex];
		return true;
	}
	float getSDFvalue(const DataIndex& dI, float& sdf) const
	{
		assert(dI() < mMcVertexSdf.size());
		return mMcVertexSdf[dI()];
	}

	void updateSurfaceParticles();

	inline bool getSDFvalue(const Eigen::Vector3li& c, float& sdf) const
	{
		return getSDFvalue(CellIndex(cellIndex(c), *this), sdf);
	}
	inline bool getSDFvalue(size_t i, size_t j, size_t k, float& sdf) const
	{
		return getSDFvalue(Eigen::Vector3li(i, j, k), sdf);
	}

	inline bool getSDFvalue(size_t cellInd, float& sdf) const
	{
		return getSDFvalue(cell(cellInd), sdf);
	}

	std::vector<Eigen::Vector3d> getTriangles() const;
	///Calculate cell indeces of neighbour cells
	std::vector<Eigen::Vector3li> getNeighbourCells(const Eigen::Vector3d& position, float radius, bool existing = true) const;
	std::vector<std::pair<size_t, std::array<std::array<int, 3>, 5>>> computeIntersectionCells() const;
	std::unordered_set<size_t> computeIntersectionCellVertices(int neighborsCnt = 0) const;
	std::unordered_set<size_t> computeIntersectionVertices(int neighbors = 0) const;
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
	inline Eigen::Vector3li cell(const Eigen::Vector3d& vec) const
	{
		float xf = (vec(0) - mLowerCorner(0)) / (mUpperCorner(0) - mLowerCorner(0));
		float yf = (vec(1) - mLowerCorner(1)) / (mUpperCorner(1) - mLowerCorner(1));
		float zf = (vec(2) - mLowerCorner(2)) / (mUpperCorner(2) - mLowerCorner(2));
		return Eigen::Vector3li(std::floor(mDimentions(0) * xf),
							   std::floor(mDimentions(1) * yf),
							   std::floor(mDimentions(2) * zf));
	
	}
	
	inline Eigen::Vector3li cell(size_t index) const
	{
		size_t k = index % mDimentions(2);
		size_t j = (index / mDimentions(2)) % mDimentions(1);
		size_t i = ((index / mDimentions(2)) / mDimentions(1));
		return Eigen::Vector3li(i, j, k);
	}
	
	///Calculate cell vertice coordinate given cell indecis
	inline Eigen::Vector3d cellCoord(const Eigen::Vector3li& vec) const
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
	inline size_t cellIndex(const Eigen::Vector3li& ind) const
	{
		return ind(0) * mDimentions(1) * mDimentions(2) + 
				ind(1) * mDimentions(2) + 
				ind(2);
	}

	inline bool getCurvature(const CellIndex& cI, Real& curvature) const
	{
		auto dI = *cI;
		if(dI == InvPrt)
			return false;
		curvature = mMcVertexCurvature[dI];
		return true;
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
		MarchingCubes(other)
	{}

protected:
	void updateGrid() override;
	void updateLevelSet() override;
	
//	bool getSDFvalue(size_t i, size_t j, size_t k, float& sdf) const override
//	{
//		auto val = mLevelSetFunction.find(cellIndex(Eigen::Vector3li(i, j, k)));
//		if(val  == mLevelSetFunction.end())
//			return false;
//		sdf = val->second;
//		return true;
//	}
};
