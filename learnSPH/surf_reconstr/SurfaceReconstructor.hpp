#pragma once
#include <Eigen/Dense>

class SurfaceReconstructor
{
	
public:
	virtual void updateLevelSet() = 0;
	virtual void updateGrid() = 0;
	virtual vector<Eigen::Vector3d> getTriangles() const = 0;
};
