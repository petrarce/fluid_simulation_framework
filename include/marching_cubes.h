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
        size_t cubesX, cubesY, cubesZ;

        vector<Real> gridPointImplicitFuncs;
        vector<Vector3R> gridPointPositions;

        Vector3R alignedCuberResolution;
        bool objectDefined = false;

    public:
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

		Object3D(const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol)
        {
            Vector3R distVec = uCorner - lCorner;

            alignedCuberResolution(0) = distVec(0) / (int(distVec(0) / cbResol(0)) + 1);
            alignedCuberResolution(1) = distVec(1) / (int(distVec(1) / cbResol(1)) + 1);
            alignedCuberResolution(2) = distVec(2) / (int(distVec(2) / cbResol(2)) + 1);

            cubesX = int(distVec(0) / cbResol(0)) + 1;
            cubesY = int(distVec(1) / cbResol(1)) + 1;
            cubesZ = int(distVec(2) / cbResol(2)) + 1;

            Vector3R curCubePosition;

            for(size_t i = 0; i < cubesX; i++) {

                curCubePosition(0) = lCorner(0) + i * alignedCuberResolution(0);

                for (size_t j = 0; j < cubesY; j++) {

                    curCubePosition(1) = lCorner(1) + j * alignedCuberResolution(1);

                    for (size_t k = 0; k < cubesZ; k++) {

                        curCubePosition(2) = lCorner(2) + k * alignedCuberResolution(2);

                        gridPointPositions.push_back(curCubePosition);
                    }
                }
            }
        };
};

class Sphere : public Object3D
{
    protected:
        Vector3R center;

    public:
        Sphere(const Real radius, const Vector3R& center, const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol):Object3D(lCorner, uCorner, cbResol)
        {
            for(size_t i = 0; i < cubesX; i++) {

                for (size_t j = 0; j < cubesY; j++) {

                    for (size_t k = 0; k < cubesZ; k++) {

                        auto curCubePosition = gridPointPositions[i * cubesY * cubesZ + j * cubesZ + k];
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
    public:
        Thorus(Real r_major, Real r_minor, Vector3R center, const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol):Object3D(lCorner, uCorner, cbResol)
        {
            for(size_t i = 0; i < cubesX; i++) {

                for (size_t j = 0; j < cubesY; j++) {

                    for (size_t k = 0; k < cubesZ; k++) {

                        auto posVec = gridPointPositions[i * cubesY * cubesZ + j * cubesZ + k] - center;
                        
                        gridPointImplicitFuncs.push_back(pow2(r_minor) - pow2(sqrt(pow2(posVec(0)) + pow2(posVec(1))) - r_major) - pow2(posVec(2)));
                    }
                }
            }
            objectDefined = true;
		};

		~Thorus(){};
};

class Fluid : public Object3D{

    public:
        Fluid(learnSPH::FluidSystem *fluidParticles, Real initValue, const Vector3R &lCorner, const Vector3R &uCorner, const Vector3R &cbResol):Object3D(lCorner, uCorner, cbResol)
        {
            gridPointImplicitFuncs.assign(cubesX * cubesY * cubesZ, -initValue);

            NeighborhoodSearch ns(fluidParticles->getCompactSupport());

            ns.add_point_set((Real*)(fluidParticles->getPositions().data()), fluidParticles->size());
            ns.add_point_set((Real*)(gridPointPositions.data()), gridPointPositions.size());

            ns.update_point_sets();

            vector<vector<unsigned int> > neighbors;

            auto fluidPositions = fluidParticles->getPositions();
            auto fluidDensities = fluidParticles->getDensities();

            for(unsigned int particleID = 0; particleID < fluidParticles->size(); particleID ++) {

                neighbors.clear();
                ns.find_neighbors(0, particleID, neighbors);

                for(unsigned int gridPointID: neighbors[1]) {

                    auto weight = kernelFunction(gridPointPositions[gridPointID], fluidPositions[particleID], fluidParticles->getSmoothingLength());

                    gridPointImplicitFuncs[gridPointID] += fluidParticles->getMass() / max(fluidDensities[particleID], fluidParticles->getRestDensity()) * weight;
                }
            }
            objectDefined = true;
        }

        Fluid(vector<Real> &params, vector<Vector3R> &positions, vector<Real> &densities, Real initValue, const Vector3R &lCorner, const Vector3R &uCorner, const Vector3R &cbResol):Object3D(lCorner, uCorner, cbResol)
        {
            gridPointImplicitFuncs.assign(cubesX * cubesY * cubesZ, -initValue);

            NeighborhoodSearch ns(params[0]);

            ns.add_point_set((Real*)(positions.data()), positions.size());
            ns.add_point_set((Real*)(gridPointPositions.data()), gridPointPositions.size());

            ns.update_point_sets();

            vector<vector<unsigned int> > neighbors;

            for(unsigned int particleID = 0; particleID < positions.size(); particleID ++) {

                neighbors.clear();
                ns.find_neighbors(0, particleID, neighbors);

                for(unsigned int gridPointID: neighbors[1]) {

                    auto weight = kernelFunction(gridPointPositions[gridPointID], positions[particleID], params[1]);

                    gridPointImplicitFuncs[gridPointID] += params[2] / max(densities[particleID], params[3]) * weight;
                }
            }
            objectDefined = true;
        }
};


namespace learnSPH
{
    class MarchingCubes {

        private:
            Object3D* object;

            Vector3R spaceLowerCorner;
            Vector3R spaceUpperCorner;
            Vector3R cubesResolution;

        public:
            void getTriangleMesh(vector<Vector3R>& triangleMesh) const;

            void setObject(Object3D* object);

            MarchingCubes(const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol);
    };
}