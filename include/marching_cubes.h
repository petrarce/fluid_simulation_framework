#pragma once
#include <iostream>
#include <types.hpp>
#include <vector>
#include <kernel.h>
#include "storage.h"

using namespace learnSPH::kernel;

using namespace std;

class Object3D
{
    protected:
        Vector3R spaceLowerCorner;
        Vector3R spaceUpperCorner;
        Vector3R cubesResolution;
        vector<Real> gridPointImplicitFuncs;
        vector<Vector3R> gridPointPositions;

        Vector3R alignedCuberResolution;
        bool objectDefined=false;
	public:
        size_t cubesX, cubesY, cubesZ;
//		virtual bool query(const Vector3R&, Real& intVal) const = 0;
		
//		virtual Vector3R lerp(const Vector3R&, const Real, const Vector3R&, const Real) const = 0;

        bool query(const size_t x, const size_t y, const size_t z) const{
            assert(objectDefined);
            return (gridPointImplicitFuncs[x*cubesY*cubesZ + y*cubesZ + z] < 0);
        }

        Vector3R interpolate(const size_t x1, const size_t y1, const size_t z1,
                             const size_t x2, const size_t y2, const size_t z2) const{
            assert(objectDefined);
            Vector3R pt1 = gridPointPositions[x1*cubesY*cubesZ + y1*cubesZ + z1];
            Vector3R pt2 = gridPointPositions[x2*cubesY*cubesZ + y2*cubesZ + z2];
            Real val1 = gridPointImplicitFuncs[x1*cubesY*cubesZ + y1*cubesZ + z1];
            Real val2 = gridPointImplicitFuncs[x2*cubesY*cubesZ + y2*cubesZ + z2];
            Vector3R distVec = (pt1 - pt2);

//            Real relation = fabs(val2) / (fabs(val1) + fabs(val2));
            Real relation = val1 / (-val2 + val1);

            if (relation <0 || relation > 1) std::cout<<relation<<std::endl;
//            assert(relation>=0 && relation<=1);
            return pt1*(1-relation) + pt2*relation;
//            return pt2 + distVec * relation;
        }

		Object3D(const Vector3R& loverCorner, const Vector3R& upperCorner, const Vector3R& cbResol):
		spaceLowerCorner(loverCorner), spaceUpperCorner(upperCorner), cubesResolution(cbResol){
            Vector3R distVec = spaceUpperCorner - spaceLowerCorner;

            alignedCuberResolution(0) = distVec(0) / (int(distVec(0) / this->cubesResolution(0)) + 1);
            alignedCuberResolution(1) = distVec(1) / (int(distVec(1) / this->cubesResolution(1)) + 1);
            alignedCuberResolution(2) = distVec(2) / (int(distVec(2) / this->cubesResolution(2)) + 1);

            cubesX = int(distVec(0) / this->cubesResolution(0)) + 1;
            cubesY = int(distVec(1) / this->cubesResolution(1)) + 1;
            cubesZ = int(distVec(2) / this->cubesResolution(2)) + 1;

            Vector3R curCubePosition = this->spaceLowerCorner;

            for(size_t i = 0; i < cubesX; i++) {

                curCubePosition(0) = this->spaceLowerCorner(0) + i * alignedCuberResolution(0);

                for (size_t j = 0; j < cubesY; j++) {

                    curCubePosition(1) = this->spaceLowerCorner(1) + j * alignedCuberResolution(1);

                    for (size_t k = 0; k < cubesZ; k++) {

                        curCubePosition(2) = this->spaceLowerCorner(2) + k * alignedCuberResolution(2);

                        gridPointPositions.push_back(curCubePosition);
                    }
                }
            }
		};
};

class Sphere : public Object3D
{
	protected:
		Real radius;
		Vector3R center;

	public:
//		virtual bool query(const Vector3R& pt, Real& intVal) const
//		{
//		    intVal = (center - pt).squaredNorm() - radius * radius;
//			return intVal < 0;
//		};


		virtual Vector3R lerp(const Vector3R& pt1, const Real val1, const Vector3R& pt2, const Real val2) const
		{
			Vector3R distVec = (pt1 - pt2);

			Real relation = fabs(val2) / (fabs(val1) + fabs(val2));

			return pt2 + distVec * relation;
		};

		Sphere(const Real rad, const Vector3R& cntr, const Vector3R& loverCorner,
		        const Vector3R& upperCorner, const Vector3R& cbResol):
		        radius(rad), center(cntr),  Object3D(loverCorner, upperCorner, cbResol){

            Vector3R curCubePosition = this->spaceLowerCorner;
            for(size_t i = 0; i < cubesX; i++) {
                for (size_t j = 0; j < cubesY; j++) {
                    for (size_t k = 0; k < cubesZ; k++) {

                        curCubePosition = gridPointPositions[i*cubesY*cubesZ + j*cubesZ + k];
                        gridPointImplicitFuncs.push_back((center - curCubePosition).squaredNorm() - radius * radius);
                    }
                }
            }
            objectDefined = true;
		};

		virtual ~Sphere(){};
};

class Thorus : public Object3D
{
	protected:
		Real rMaj;
		Real rMin;
		Vector3R center;

	public:
//		virtual bool query(const Vector3R& pt, Real& intVal) const
//		{
//			Vector3R posVec = pt - center;
//
//			intVal = pow2(rMin) - pow2(sqrt(pow2(posVec(0)) + pow2(posVec(1))) - rMaj) - pow2(posVec(2));
//
//			return intVal < 0;
//		};

		virtual Vector3R lerp(const Vector3R& pt1, const Real val1, const Vector3R& pt2, const Real val2) const
		{
			Vector3R distVec = (pt1 - pt2);

			Real relation = fabs(val2) / (fabs(val1) + fabs(val2));

			return pt2 + distVec * relation;
		};

		Thorus(Real rmj, Real rmn, Vector3R cntr, const Vector3R& loverCorner,
               const Vector3R& upperCorner, const Vector3R& cbResol):
               rMaj(rmj), rMin(rmn), center(cntr),  Object3D(loverCorner, upperCorner, cbResol){

		    Vector3R curCubePosition = this->spaceLowerCorner;
            for(size_t i = 0; i < cubesX; i++) {
                for (size_t j = 0; j < cubesY; j++) {
                    for (size_t k = 0; k < cubesZ; k++) {

                        curCubePosition = gridPointPositions[i*cubesY*cubesZ + j*cubesZ + k];
                        Vector3R posVec = curCubePosition - center;
                        gridPointImplicitFuncs.push_back(pow2(rMin) - pow2(sqrt(pow2(posVec(0)) + pow2(posVec(1))) - rMaj) - pow2(posVec(2)));
                    }
                }
            }
            objectDefined = true;
		};

		~Thorus(){};
};

class GeneralShape : public Object3D{
    protected:
        learnSPH::FluidSystem* fluidParticles;
        Real initValue;
    public:
        GeneralShape(learnSPH::FluidSystem* particleSet, Real initValue, const Vector3R& loverCorner,
                     const Vector3R& upperCorner, const Vector3R& cbResol):
                     fluidParticles(particleSet), initValue(initValue), Object3D(loverCorner, upperCorner, cbResol){
            gridPointImplicitFuncs.clear();
            for(size_t i = 0; i < cubesX; i++) {
                for (size_t j = 0; j < cubesY; j++) {
                    for (size_t k = 0; k < cubesZ; k++) {
                        gridPointImplicitFuncs.push_back(-initValue);
                    }}}
            NeighborhoodSearch ns(fluidParticles->getCompactSupport());
            unsigned int particleSetID = ns.add_point_set((Real*)(fluidParticles->getPositions().data()), fluidParticles->size());
            unsigned int verticeSetID = ns.add_point_set((Real*)(gridPointPositions.data()), gridPointPositions.size());
            ns.update_point_sets();
            vector<vector<unsigned int>> neighbors;
            for(unsigned int particleID=0; particleID < fluidParticles->size(); particleID++){
                neighbors.clear();
                ns.find_neighbors(particleSetID, particleID, neighbors);
                for(unsigned int gridPointID: neighbors[verticeSetID]){
                    gridPointImplicitFuncs[gridPointID] += fluidParticles->getMass()/
                            fluidParticles->getDensities()[particleID] *
                            kernelFunction(gridPointPositions[gridPointID], fluidParticles->getPositions()[particleID], fluidParticles->getCompactSupport());
                }

            }
            objectDefined = true;
        }
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
