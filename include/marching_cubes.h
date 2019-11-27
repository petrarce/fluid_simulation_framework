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
        bool objectDefined = false;

	public:
        size_t cubesX, cubesY, cubesZ;

        bool query(const size_t x, const size_t y, const size_t z) const
        {
            assert(objectDefined);
            return (gridPointImplicitFuncs[x * cubesY * cubesZ + y * cubesZ + z] < 0);
        }

        Vector3R interpolate(const size_t x1, const size_t y1, const size_t z1, const size_t x2, const size_t y2, const size_t z2) const
        {
            assert(objectDefined);
            
            Vector3R pt1 = gridPointPositions[x1 * cubesY * cubesZ + y1 * cubesZ + z1];
            Vector3R pt2 = gridPointPositions[x2 * cubesY * cubesZ + y2 * cubesZ + z2];

            Real val1 = gridPointImplicitFuncs[x1 * cubesY * cubesZ + y1 * cubesZ + z1];
            Real val2 = gridPointImplicitFuncs[x2 * cubesY * cubesZ + y2 * cubesZ + z2];
            
            Vector3R distVec = (pt1 - pt2);

            Real relation = val1 / (-val2 + val1);

            assert(0.0 <= relation && relation <= 1.0);

            return pt1 * (1.0 - relation) + pt2 * relation;
        }

		Object3D(const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol):spaceLowerCorner(lCorner), spaceUpperCorner(uCorner), cubesResolution(cbResol)
        {
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
		Sphere(const Real rad, const Vector3R& cntr, const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol):radius(rad), center(cntr), Object3D(lCorner, uCorner, cbResol)
        {
            Vector3R curCubePosition = this->spaceLowerCorner;

            for(size_t i = 0; i < cubesX; i++) {
                for (size_t j = 0; j < cubesY; j++) {
                    for (size_t k = 0; k < cubesZ; k++) {

                        curCubePosition = gridPointPositions[i * cubesY * cubesZ + j * cubesZ + k];
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
		Real r_a;
		Real r_b;
		Vector3R center;

	public:
		Thorus(Real r_A, Real r_B, Vector3R cntr, const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol):r_a(r_A), r_b(r_B), center(cntr), Object3D(lCorner, uCorner, cbResol)
        {
		    Vector3R curCubePosition = this->spaceLowerCorner;

            for(size_t i = 0; i < cubesX; i++) {
                for (size_t j = 0; j < cubesY; j++) {
                    for (size_t k = 0; k < cubesZ; k++) {

                        curCubePosition = gridPointPositions[i * cubesY * cubesZ + j * cubesZ + k];
                        Vector3R posVec = curCubePosition - center;
                        gridPointImplicitFuncs.push_back(pow2(r_b) - pow2(sqrt(pow2(posVec(0)) + pow2(posVec(1))) - r_a) - pow2(posVec(2)));
                    }
                }
            }
            objectDefined = true;
		};

		~Thorus(){};
};

class GeneralShape : public Object3D{
    protected:
        learnSPH::FluidSystem * fluidParticles;
        Real initValue;

    public:
        GeneralShape(learnSPH::FluidSystem *particleSet, Real initValue, const Vector3R &lCorner, const Vector3R &uCorner, const Vector3R &cbResol):fluidParticles(particleSet), initValue(initValue), Object3D(lCorner, uCorner, cbResol)
        {
            gridPointImplicitFuncs.assign(cubesX * cubesY * cubesZ, -initValue);

            NeighborhoodSearch ns(fluidParticles->getCompactSupport());

            unsigned int particleSetID = ns.add_point_set((Real*)(fluidParticles->getPositions().data()), fluidParticles->size());
            unsigned int verticeSetID = ns.add_point_set((Real*)(gridPointPositions.data()), gridPointPositions.size());
            
            ns.update_point_sets();

            vector<vector<unsigned int> > neighbors;

            for(unsigned int particleID = 0; particleID < fluidParticles->size(); particleID ++) {

                neighbors.clear();
                ns.find_neighbors(particleSetID, particleID, neighbors);

                for(unsigned int gridPointID: neighbors[verticeSetID]) {

                    auto weight = kernelFunction(gridPointPositions[gridPointID], fluidParticles->getPositions()[particleID], fluidParticles->getCompactSupport());

                    gridPointImplicitFuncs[gridPointID] += fluidParticles->getMass() / fluidParticles->getRestDensity() * weight;
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

			opcode init(const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol);

			MarchingCubes();

			~MarchingCubes();
	};
}