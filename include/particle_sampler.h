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
	void sample_triangle(const Vector3R &vertex_a, const Vector3R &vertex_b, const Vector3R &vertex_c, Real samplingDistance, vector<Vector3R> &borderParticles);
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
	BorderSystem* sample_border_box(const Vector3R &lowerCorner, const Vector3R &upperCorner, Real restDensity, Real samplingDistance, Real eta);
	
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
	 *	samples border particles on the surface of a sphere
	 *	Args:
	 *		radius				- radius of the sphere
	 *		center				- center of the sphere
	 *		restDensity			- rest density for the border (assumed as constant among all particles)
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 *		eta					- eta for the border (assumed as constant among all particles)
	 */
	BorderSystem* sample_border_sphere(Real radius, const Vector3R &center, Real restDensity, Real samplingDistance, Real eta);

	Real sample_cube(vector<Vector3R> &positions, const Vector3R &lowerCorner, const Vector3R &upperCorner, Real samplingDistance);
	/*
	 *	samples fluid particles inside a cube in axis-aligned fashion.
	 *	Args:
	 *		lowerA				- lower corner of the cube A
	 *		upperA				- upper corner of the cube A
	 *		restDensity			- rest density for the fluid (assumed as constant among all particles)
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 */
	FluidSystem* single_dam(const Vector3R &lowerA, const Vector3R &upperA, Real restDensity, Real samplingDistance, Real eta);
	/*
	 *	samples fluid particles inside two cubes in axis-aligned fashion.
	 *	Args:
	 *		lowerA				- lower corner of the cube A
	 *		upperA				- upper corner of the cube A
	 *		lowerB				- lower corner of the cube B
	 *		upperB				- upper corner of the cube B
	 *		restDensity			- rest density for the fluid (assumed as constant among all particles)
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 */
	FluidSystem* double_dam(const Vector3R &lowerA, const Vector3R &upperA, const Vector3R &lowerB, const Vector3R &upperB, Real restDensity, Real samplingDistance, Real eta);

	void sample_shell(vector<Vector3R> &positions, const Vector3R &center, Real radius, Real samplingDistance);
	/*
	 *	samples fluid particles inside a sphere with initial velocity
	 *	Args:
	 *		center				- center of the sphere
	 *		speed				- initial velocity
	 *		radius				- radius of the sphere
	 *		restDensity			- rest density for the fluid (assumed as constant among all particles)
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 */
	FluidSystem* water_drop(const Vector3R &center, const Vector3R &speed, Real radius, Real restDensity, Real samplingDistance, Real eta);
};
