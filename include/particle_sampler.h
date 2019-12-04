#pragma once
#include <storage.h>

using namespace std;

namespace learnSPH{
	/*
	 *	samples fluid particles on a triangular face
	 *	Args:
	 *		corner_a			- corner A of the triangular face
	 *		corner_b			- corner B of the triangular face
	 *		corner_c			- corner C of the triangular face
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 *		borderParticles		- container for fetched particles
	 *		hexagonal			- whether to employ hexagonal patterns when sampling the particles (default: false)
	 */
	void sample_border_face(const Vector3R& vertex_a, 
							const Vector3R& vertex_b, 
							const Vector3R& vertex_c, 
							const Real samplingDistance, 
							vector<Vector3R>& borderParticles, 
							bool hexagonal);
	/*
	 *	samples fluid particles on a triangular face
	 *	Args:
	 *		corner_a			- corner A of the triangular face
	 *		corner_b			- corner B of the triangular face
	 *		corner_c			- corner C of the triangular face
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 *		borderParticles		- container for fetched particles
	 *		hexagonal			- whether to employ hexagonal patterns when sampling the particles (default: false)
	 */
	void sample_box_face(const Vector3R& lowerCorner, 
									const Vector3R& upperCorner, 
									const Real samplingDistance, 
									vector<Vector3R>& borderParticles, 
									bool hexagonal);
	/*
	 *	samples border particles on the faces of a cube
	 *	Args:
	 *		lowerCircleRadius			- radius of lower bound of the cone
	 *		lowerCircleCenter			- point where lower bound of circle center is situated
	 *		upperCircleRadius			- radius of upper bound of the cone
	 *		upperCircleCenter			- point where upper bound of circle center is situated
	 *		samplingDistance			- distance between the centers of each two adjacent particles
	 */
	void sample_border_cone_face(		const Real lowerCircleRadius, 
										const Vector3R& lowerCircleCenter,
										const Real upperCircleRadius,
										const Vector3R& upperCircleCenter,
										Real samplingDistance,
										vector<Vector3R>& inpBorderParticles);
	/*
	 *	samples fluid particles inside a cube in axis-aligned fashion.
	 *	Args:
	 *		upperCorner			- upper corner of the cube
	 *		lowerCorner			- lower corner of the cube
	 *		restDensity			- rest density for the fluid (assumed as constant among all particles)
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 */
	FluidSystem* sample_fluid_cube(const Vector3R& lowerCorner, 
									const Vector3R& upperCorner, 
									const Real restDensity, 
									const Real samplingDistance, 
									const Real eta);
	/*
	 *	samples border particles on the faces of a cube
	 *	Args:
	 *		upperCorner			- upper corner of the cube
	 *		lowerCorner			- lower corner of the cube
	 *		restDensity			- rest density for the border (assumed as constant among all particles)
	 *		samplingDistance	- distance between the centers of each two adjacent particles
	 *		hexagonal			- whether to employ hexagonal patterns when sampling the particles (default: false)
	 *		diameter			- particle diameter for the border (assumed as constant among all particles)
	 */
	BorderSystem* sample_border_box(const Vector3R& lowerCorner, 
									const Vector3R& upperCorner, 
									const Real restDensity, 
									const Real samplingDistance, 
									const Real eta, 
									const bool hexagonal);
	/*
	 *	samples border particles on the faces of a cube
	 *	Args:
	 *		lowerCircleRadius			- radius of lower bound of the cone
	 *		lowerCircleCenter			- point where lower bound of circle center is situated
	 *		upperCircleRadius			- radius of upper bound of the cone
	 *		upperCircleCenter			- point where upper bound of circle center is situated
	 *		samplingDistance			- distance between the centers of each two adjacent particles
	 *		eta							- eta sampling distance factor - used for computing compact support of border partcles
	 		restDensity					- rest density for the border (assumed as constant among all particles)
	 */
	BorderSystem* sample_border_cone(	const Real lowerCircleRadius, 
										const Vector3R& lowerCircleCenter,
										const Real upperCircleRadius,
										const Vector3R& upperCircleCenter,
										Real restDensity,
										Real samplingDistance,
										Real eta);

	BorderSystem* sample_border_cone_and_box(	const Real lowerCircleRadius, 
										const Vector3R& lowerCircleCenter,
										const Real upperCircleRadius,
										const Vector3R& upperCircleCenter,
										const Vector3R& lowerCorner, 
										const Vector3R& upperCorner,
										Real restDensity,
										Real samplingDistance,
										Real eta,
										bool hexagonal);
};
