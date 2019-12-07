#pragma once
#include <storage.h>

using namespace std;

namespace learnSPH{
	/*
	 *	samples border particles on a triangular face
	 *	Args:
	 *		vertex_a			- corner A of the triangular face
	 *		vertex_b			- corner B of the triangular face
	 *		vertex_c			- corner C of the triangular face
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 *		borderParticles		- container for fetched particles
	 *		hexagonal			- whether to employ hexagonal patterns when sampling the particles (default: false)
	 */
	void sample_triangle(const Vector3R &vertex_a, const Vector3R &vertex_b, const Vector3R &vertex_c, Real samplingDistance, vector<Vector3R> &borderParticles, bool hexagonal);
	/*
	 *	samples border particles on the faces of a cube
	 *	Args:
	 *		lowerCorner			- lower corner of the cube
	 *		upperCorner			- upper corner of the cube
	 *		restDensity			- rest density for the border (assumed as constant among all particles)
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 *		eta					- eta for the border (assumed as constant among all particles)
	 *		hexagonal			- whether to employ hexagonal patterns when sampling the particles (default: false)
	 */
	BorderSystem* sample_border_box(const Vector3R &lowerCorner, const Vector3R &upperCorner, Real restDensity, Real samplingDistance, Real eta, bool hexagonal);
	
	void sample_ring(vector<Vector3R> &borderParticles, const Vector3R &center, Real radius, const Vector3R &unit_x, const Vector3R &unit_y, Real samplingDistance);
	/*
	 *	samples border particles on the surface of a cone
	 *	Args:
	 *		lowerRadius			- radius of the lower cross section of the truncated cone
	 *		lowerCenter			- center of the lower cross section of the truncated cone
	 *		upperRadius			- radius of the upper cross section of the truncated cone
	 *		upperCenter			- center of the upper cross section of the truncated cone
	 *		restDensity			- rest density for the border (assumed as constant among all particles)
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 *		eta					- eta for the border (assumed as constant among all particles)
	 */
	BorderSystem* sample_border_cone(Real lowerRadius, const Vector3R &lowerCenter, Real upperRadius, const Vector3R &upperCenter, Real restDensity, Real samplingDistance, Real eta);
	/*
	 *	samples fluid particles inside a cube in axis-aligned fashion.
	 *	Args:
	 *		upperCorner			- upper corner of the cube
	 *		lowerCorner			- lower corner of the cube
	 *		restDensity			- rest density for the fluid (assumed as constant among all particles)
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 */
	FluidSystem* sample_fluid_cube(const Vector3R &lowerCorner, const Vector3R &upperCorner, Real restDensity, Real samplingDistance, Real eta);
	
	void sample_sphere(vector<Vector3R>& borderParticles, const Real radius, const Vector3R center, const Real samplingDistance);
};
