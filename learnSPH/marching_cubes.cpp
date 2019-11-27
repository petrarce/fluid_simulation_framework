#include <iostream>
#include <cassert>
#include <marching_cubes.h>
#include <look_up_tables.hpp>

using namespace std;

opcode learnSPH::MarchingCubes::getTriangleMesh(vector<Vector3R>& triangleMesh) const
{
	assert(this->obj3D != NULL);

	Vector3R distVec = this->spaceUpperCorner - this->spaceLowerCorner;

	size_t cubesX = int(distVec(0) / this->cubesResolution(0)) + 1;
	size_t cubesY = int(distVec(1) / this->cubesResolution(1)) + 1;
	size_t cubesZ = int(distVec(2) / this->cubesResolution(2)) + 1;

	triangleMesh.clear();
	triangleMesh.reserve(cubesX * cubesY * cubesZ * 12);

	for(size_t i = 0; i < cubesX - 1; i++) {

		for(size_t j = 0; j < cubesY - 1; j++) {

			for(size_t k = 0; k < cubesZ - 1; k++) {

				std::array<bool, 8> ptsConfig;

				for(int l = 0; l < 8; l++) ptsConfig[l] = this->obj3D->query(i + CELL_VERTICES[l][0], j + CELL_VERTICES[l][1], k + CELL_VERTICES[l][2]);

				std::array<std::array<int, 3>, 5> triangle_type = getMarchingCubesCellTriangulation(ptsConfig);

				for(size_t l = 0; l < 5; l++) {
                    
					for(size_t m = 0; m < 3; m++) {
						if(triangle_type[l][m] == -1) break;

						Vector3R interpolatedPt = this->obj3D->interpolate(
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
