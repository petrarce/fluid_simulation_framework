#pragma once
#include <storage.h>

using namespace std;


namespace learnSPH{
	typedef struct FaceVertexList_t{
		vector<Vector3R> vecrtexes;
		vector<Vector3i> faceIndexes;
	} FaceVertexList;

	/*
		Parses and puts data from wavefront obj file to FaceVertexList data structure
		path_to_wavefront - path to wavefront obj file model
	*/
	FaceVertexList* parse_wavefront(string path_to_wavefront);
	/*
	 *	samples points according to triangle mesh, provided by wavefront.obj model
	 *	Args:
	 * 		borderParticles - vector, which finally be deployed with sampled particles
	 *		transitionMatr  - transformation matrix, according to which model points will be modified
	 *		patToModel		- path to model file 
	 *		samplingDistance- maximal sampling distance between the points	
	*/
   	void sample_border_model_surface(vector<Vector3R>& borderParticles, const Matrix4d transitionMatr, const string& patToModel, Real samplingDistance);
	BorderSystem* sample_border_model(const Matrix4d& transitionMatr, const string& patToModel, Real restDensity, Real samplingDistance, Real eta);

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
