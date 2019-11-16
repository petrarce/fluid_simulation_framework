#pragma once
#include <iostream>
#include <types.hpp>
#include <vector>

using namespace std;

class Object3D
{
public:
	virtual bool query(const Vector3R&, Real& intVal) const = 0;
	virtual Vector3R lerp(const Vector3R& , const Real , const Vector3R& , const Real ) const = 0;
};

class Sphere : public Object3D
{
	Real radius;
	Vector3R center;
public:
	virtual bool query(const Vector3R& pt, Real& intVal) const
	{
		intVal = (center - pt).squaredNorm() - radius*radius;
		if(intVal < 0){
			return true;
		} 
		return false;
	};

	virtual Vector3R lerp(const Vector3R& pt1, const Real val1, const Vector3R& pt2, const Real val2) const
	{
		Vector3R distVec = (pt1 - pt2);
		Real relation = fabs(val2/(fabs(val1)+fabs(val2)));
		pr_dbg("relation = %f", relation);
		Vector3R targPt = pt2 + distVec * relation;
		return targPt;
	};

	Sphere(const Real rad,const Vector3R& cntr):
		radius(rad),
		center(cntr){};
	virtual ~Sphere(){};
};

namespace learnSPH
{

	class MarchingCubes {
	private:
		const Object3D* obj3D;
		Vector3R spaceLowerCorner;
		Vector3R spaceUpperCorner;
		Vector3R cubesResolution;


	public:

		opcode getTriangleMesh(vector<Vector3R>& triangleMesh) const;
		opcode setObject(const Object3D* const obj);
		opcode init(const Vector3R& loverCorner, const Vector3R& upperCorner, const Vector3R& cbResol);
		MarchingCubes();
		~MarchingCubes();
	};
}