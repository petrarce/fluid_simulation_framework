#pragma once
#include <iostream>
#include <cmath>
#include <types.hpp>
#include <vector>
#include <learnSPH/core/kernel.h>
#include <storage.h>

using namespace learnSPH::kernel;

using namespace std;

class Object3D
{
    protected:
        bool objectDefined = false;

        vector<Real> gridPointImplicitFuncs;
        vector<Vector3R> gridPointPositions;

        Vector3R unit_march_vec;

    public:
        size_t cubesX, cubesY, cubesZ;

        Vector3R lowerCorner;
        Vector3R upperCorner;

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
            this->lowerCorner = lCorner;
            this->upperCorner = uCorner;

            Vector3R distVec = uCorner - lCorner;

            this->cubesX = ceil(distVec(0) / cbResol(0));
            this->cubesY = ceil(distVec(1) / cbResol(1));
            this->cubesZ = ceil(distVec(2) / cbResol(2));

            unit_march_vec(0) = distVec(0) / (this->cubesX - 1);
            unit_march_vec(1) = distVec(1) / (this->cubesY - 1);
            unit_march_vec(2) = distVec(2) / (this->cubesZ - 1);

            Vector3R curCubePosition;

            for(size_t i = 0; i < cubesX; i++) {

                curCubePosition(0) = lCorner(0) + i * unit_march_vec(0);

                for (size_t j = 0; j < cubesY; j++) {

                    curCubePosition(1) = lCorner(1) + j * unit_march_vec(1);

                    for (size_t k = 0; k < cubesZ; k++) {

                        curCubePosition(2) = lCorner(2) + k * unit_march_vec(2);

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

            auto positions = fluidParticles->getPositions();
            auto densities = fluidParticles->getDensities();

            for (size_t particleID = 0; particleID < positions.size(); particleID ++) {

                auto offset = positions[particleID] - lowerCorner;

                int grid_x_near = floor((offset(0) - fluidParticles->getCompactSupport()) / unit_march_vec(0));
                int grid_y_near = floor((offset(1) - fluidParticles->getCompactSupport()) / unit_march_vec(1));
                int grid_z_near = floor((offset(2) - fluidParticles->getCompactSupport()) / unit_march_vec(2));

                grid_x_near = std::max(grid_x_near, 0);
                grid_y_near = std::max(grid_y_near, 0);
                grid_z_near = std::max(grid_z_near, 0);

                int grid_x_far = ceil((offset(0) + fluidParticles->getCompactSupport()) / unit_march_vec(0));
                int grid_y_far = ceil((offset(1) + fluidParticles->getCompactSupport()) / unit_march_vec(1));
                int grid_z_far = ceil((offset(2) + fluidParticles->getCompactSupport()) / unit_march_vec(2));

                grid_x_far = std::min(grid_x_far, int(cubesX - 1));
                grid_y_far = std::min(grid_y_far, int(cubesY - 1));
                grid_z_far = std::min(grid_z_far, int(cubesZ - 1));

                for (size_t x = grid_x_near; x <= grid_x_far; x ++) {

                    for (size_t y = grid_y_near; y <= grid_y_far; y ++) {

                        for (size_t z = grid_z_near; z <= grid_z_far; z ++) {

                            auto grid_idx = x * cubesY * cubesZ + y * cubesZ + z;

                            assert(grid_idx < gridPointPositions.size());

                            auto weight = kernelFunction(gridPointPositions[grid_idx], positions[particleID], fluidParticles->getSmoothingLength());

                            gridPointImplicitFuncs[grid_idx] += fluidParticles->getMass() / max(densities[particleID], fluidParticles->getRestDensity()) * weight;
                        }
                    }
                }
            }
            objectDefined = true;
        }

        Fluid(vector<Real> &params, vector<Vector3R> &positions, vector<Real> &densities, Real initValue, const Vector3R &lCorner, const Vector3R &uCorner, const Vector3R &cbResol):Object3D(lCorner, uCorner, cbResol)
        {
            gridPointImplicitFuncs.assign(cubesX * cubesY * cubesZ, -initValue);

            for (size_t particleID = 0; particleID < positions.size(); particleID ++) {

                auto offset = positions[particleID] - lowerCorner;

                int grid_x_near = floor((offset(0) - params[0]) / unit_march_vec(0));
                int grid_y_near = floor((offset(1) - params[0]) / unit_march_vec(1));
                int grid_z_near = floor((offset(2) - params[0]) / unit_march_vec(2));

                grid_x_near = std::max(grid_x_near, 0);
                grid_y_near = std::max(grid_y_near, 0);
                grid_z_near = std::max(grid_z_near, 0);

                int grid_x_far = ceil((offset(0) + params[0]) / unit_march_vec(0));
                int grid_y_far = ceil((offset(1) + params[0]) / unit_march_vec(1));
                int grid_z_far = ceil((offset(2) + params[0]) / unit_march_vec(2));

                grid_x_far = std::min(grid_x_far, int(cubesX - 1));
                grid_y_far = std::min(grid_y_far, int(cubesY - 1));
                grid_z_far = std::min(grid_z_far, int(cubesZ - 1));

                for (size_t x = grid_x_near; x <= grid_x_far; x ++) {

                    for (size_t y = grid_y_near; y <= grid_y_far; y ++) {

                        for (size_t z = grid_z_near; z <= grid_z_far; z ++) {

                            auto grid_idx = x * cubesY * cubesZ + y * cubesZ + z;

                            assert(grid_idx < gridPointPositions.size());

                            auto weight = kernelFunction(gridPointPositions[grid_idx], positions[particleID], params[1]);

                            gridPointImplicitFuncs[grid_idx] += params[2] / max(densities[particleID], params[3]) * weight;
                        }
                    }
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

            Vector3R lowerCorner;
            Vector3R upperCorner;

            size_t cubesX;
            size_t cubesY;
            size_t cubesZ;

        public:
            void getTriangleMesh(vector<Vector3R>& triangleMesh) const;

            void setObject(Object3D* object);

            MarchingCubes(const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol);
    };
}
