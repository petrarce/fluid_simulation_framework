#include <iostream>
#include <cassert>
#include <marching_cubes.h>
#include <triangle_table.h>
#include "marching_cubes_lut.hpp"

using namespace std;

opcode learnSPH::MarchingCubes::getTriangleMesh(vector<Vector3R>& triangleMesh) const
{
	assert(this->obj3D != NULL);

	Vector3R distVec = this->spaceUpperCorner - this->spaceLowerCorner;
	Vector3R alignedCuberResolution;

	alignedCuberResolution(0) = distVec(0) / (int(distVec(0) / this->cubesResolution(0)) + 1);
	alignedCuberResolution(1) = distVec(1) / (int(distVec(1) / this->cubesResolution(1)) + 1);
	alignedCuberResolution(2) = distVec(2) / (int(distVec(2) / this->cubesResolution(2)) + 1);

	size_t cubesX = int(distVec(0) / this->cubesResolution(0)) + 1;
	size_t cubesY = int(distVec(1) / this->cubesResolution(1)) + 1;
	size_t cubesZ = int(distVec(2) / this->cubesResolution(2)) + 1;

	triangleMesh.clear();
	triangleMesh.reserve(cubesX * cubesY * cubesZ * 12);

	const Vector3R cubePtOffset[8] = {
		{0,							0,							0},
		{alignedCuberResolution(0), 0, 							0},
		{alignedCuberResolution(0), alignedCuberResolution(1), 	0},
		{0, 						alignedCuberResolution(1), 	0},
		{0,							0,							alignedCuberResolution(2)},
		{alignedCuberResolution(0), 0, 							alignedCuberResolution(2)},
		{alignedCuberResolution(0), alignedCuberResolution(1), 	alignedCuberResolution(2)},
		{0, 						alignedCuberResolution(1), 	alignedCuberResolution(2)},
	};

	const char cubePts[12][2] = {
		{0, 1},
		{1, 2},
		{2, 3},
		{3, 0},
		{4, 5},
		{5, 6},
		{6, 7},
		{7, 4},
		{0, 4},
		{1, 5},
		{2, 6},
		{3, 7}
	};

//	Vector3R curCubePosition = this->spaceLowerCorner;

	for(size_t i = 0; i < cubesX; i++) {

//		curCubePosition(0) = this->spaceLowerCorner(0) + i * alignedCuberResolution(0);

		for(size_t j = 0; j < cubesY; j++) {

//			curCubePosition(1) = this->spaceLowerCorner(1) + j * alignedCuberResolution(1);

			for(size_t k = 0; k < cubesZ; k++) {

//				curCubePosition(2) = this->spaceLowerCorner(2) + k * alignedCuberResolution(2);

//				uint8_t ptsConfig = 0;
                std::array<bool, 8> ptsConfig;
				Real interpolVal[8] = {0, 0, 0, 0, 0, 0, 0, 0};

				for(int l = 0; l < 8; l++) {
//					Vector3R cubePt = curCubePosition + cubePtOffset[l];
//					ptsConfig = ptsConfig | (this->obj3D->query(cubePt, interpolVal[l]) << l);
                    ptsConfig[l] = this->obj3D->query(i+CELL_VERTICES[l][0],
                            j+CELL_VERTICES[l][1], k + CELL_VERTICES[l][2]);
				}

//				for(size_t l = 0; l < 16; l++) {
//
//					if(TRIANGLE_TABLE[ptsConfig][l] == -1) break;
//
//					Vector3R interpolatedPt = this->obj3D->lerp(
//										curCubePosition + cubePtOffset[cubePts[TRIANGLE_TABLE[ptsConfig][l]][0]],
//										interpolVal[cubePts[TRIANGLE_TABLE[ptsConfig][l]][0]],
//										curCubePosition + cubePtOffset[cubePts[TRIANGLE_TABLE[ptsConfig][l]][1]],
//										interpolVal[cubePts[TRIANGLE_TABLE[ptsConfig][l]][1]]);
//
//					triangleMesh.push_back(interpolatedPt);
//				}
                std::array<std::array<int, 3>, 5> triangle_type = getMarchingCubesCellTriangulation(ptsConfig);
                for(size_t l = 0; l < 5; l++) {
                    bool in_border=true;
                    for(size_t m=0; m < 3; m++) {
                        if (i+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][0] < cubesX &&
                            i+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][0] < cubesX &&
                            j+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][1] < cubesY &&
                            j+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][1] < cubesY &&
                            k+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][2] < cubesZ &&
                            k+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][2] < cubesZ){}
                        else in_border=false;
                    }
                    if(in_border){
                        for(size_t m=0; m < 3; m++) {
                            if(triangle_type[l][m] == -1) break;

                            Vector3R interpolatedPt = this->obj3D->interpolate(i+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][0],
                                                                               j+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][1],
                                                                               k+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][2],
                                                                               i+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][0],
                                                                               j+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][1],
                                                                               k+CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][2]);

                            triangleMesh.push_back(interpolatedPt);}
                    }

                    }

				}
			}
		}
	return STATUS_OK;
}

opcode learnSPH::MarchingCubes::init(const Vector3R& loverCorner, const Vector3R& upperCorner, const Vector3R& cbResol)
{
	this->spaceLowerCorner = loverCorner;
	this->spaceUpperCorner = upperCorner;
	this->cubesResolution = cbResol;
	return STATUS_OK;
}

opcode learnSPH::MarchingCubes::setObject(const Object3D* const obj)
{
	assert(obj != NULL);
	this->obj3D = obj;
	return STATUS_OK;
}

learnSPH::MarchingCubes::MarchingCubes():obj3D(NULL){}

learnSPH::MarchingCubes::~MarchingCubes(){}
