#include <iostream>
#include <cassert>
#include <cmath>
#include <learnSPH/surf_reconstr/marching_cubes.h>
#include <learnSPH/surf_reconstr/look_up_tables.hpp>

using namespace std;

void learnSPH::MarchingCubes::getTriangleMesh(vector<Vector3R>& triangleMesh) const
{
	assert(this->object != NULL);

	triangleMesh.clear();
	triangleMesh.reserve(cubesX * cubesY * cubesZ * 12);

	for(size_t i = 0; i < cubesX - 1; i++) {

		for(size_t j = 0; j < cubesY - 1; j++) {

			for(size_t k = 0; k < cubesZ - 1; k++) {

				std::array<bool, 8> ptsConfig;

				for(int l = 0; l < 8; l++) ptsConfig[l] = this->object->query(i + CELL_VERTICES[l][0], j + CELL_VERTICES[l][1], k + CELL_VERTICES[l][2]);

				std::array<std::array<int, 3>, 5> triangle_type = getMarchingCubesCellTriangulation(ptsConfig);

				for(size_t l = 0; l < 5; l++) {
                    
					for(size_t m = 0; m < 3; m++) {
						if(triangle_type[l][m] == -1) break;

						Vector3R interpolatedPt = this->object->interpolate(
																		i + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][0],
																		j + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][1],
																		k + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][0]][2],
																		i + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][0],
																		j + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][1],
																		k + CELL_VERTICES[CELL_EDGES[triangle_type[l][m]][1]][2]);

						triangleMesh.push_back(interpolatedPt);
					}
				}
			}
		}
	}
}

void learnSPH::MarchingCubes::setObject(Object3D* object)
{
	assert(object != NULL);

	assert(object->lowerCorner == this->lowerCorner);
	assert(object->upperCorner == this->upperCorner);

	assert(object->cubesX == this->cubesX);
	assert(object->cubesY == this->cubesY);
	assert(object->cubesZ == this->cubesZ);

	this->object = object;
}

learnSPH::MarchingCubes::MarchingCubes(const Vector3R& lCorner, const Vector3R& uCorner, const Vector3R& cbResol)
{
	this->object = NULL;

	this->lowerCorner = lCorner;
	this->upperCorner = uCorner;

	Vector3R distVec = uCorner - lCorner;

	this->cubesX = ceil(distVec(0) / cbResol(0));
	this->cubesY = ceil(distVec(1) / cbResol(1));
	this->cubesZ = ceil(distVec(2) / cbResol(2));
}
