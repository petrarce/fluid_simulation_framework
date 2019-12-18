#include <storage.h>
#include <particle_sampler.h>
#include <kernel.h>
#include <types.hpp>
#include <Eigen/Geometry>
#include <fstream>
#include <sstream>

using namespace std;
using namespace learnSPH;
using namespace learnSPH::kernel;

FaceVertexList* learnSPH::parse_wavefront(string path_to_wavefront)
{
	ifstream inf(path_to_wavefront);
	if(!inf.is_open()){
		return nullptr;
	}

	stringstream ss;
	ss << inf.rdbuf();
	inf.close();

	string line;
	Vector3R newVertex;
	Vector3i newIndex;
	FaceVertexList* listOfVerticesAndVertIndexes = new FaceVertexList();;
	while(getline(ss, line, '\n')){
		stringstream liness(line);
		string word;
		liness >> word;
		if(word[0] == '#' ||
			!(word == "f" || word == "v")){
			continue;
		}
		string vertexOrIndex = word;
		uint8_t counter = 0;
		while(liness >> word){
			assert(counter < 3);
			if(vertexOrIndex == "v"){
				newVertex[counter] = (stod(word));
			} else {
				newIndex[counter] = (stoi(word) - 1);
			}
			counter++;
		}

		if(vertexOrIndex == "v"){
			listOfVerticesAndVertIndexes->vecrtexes.push_back(newVertex);
		} else {
			listOfVerticesAndVertIndexes->faceIndexes.push_back(newIndex);
		}
	}
	return listOfVerticesAndVertIndexes;
}


FluidSystem* learnSPH::sample_fluid_cube(const Vector3R &lowerCorner, const Vector3R &upperCorner, Real restDensity, Real samplingDistance, Real eta)
{
	
	assert(restDensity > 0.0);
	assert(samplingDistance > 0.0);
	assert((upperCorner - lowerCorner).dot(upperCorner - lowerCorner) > 0);

	Vector3R distVector = upperCorner - lowerCorner;

	size_t num_of_part_x_direction = fabs(distVector[0] / samplingDistance) + 1;
	size_t num_of_part_y_direction = fabs(distVector[1] / samplingDistance) + 1;
	size_t num_of_part_z_direction = fabs(distVector[2] / samplingDistance) + 1;

	Real delX = samplingDistance * distVector[0] / fabs(distVector[0]);
	Real delY = samplingDistance * distVector[1] / fabs(distVector[1]);
	Real delZ = samplingDistance * distVector[2] / fabs(distVector[2]);

	size_t totalNumOfPrticles = num_of_part_x_direction * num_of_part_y_direction * num_of_part_z_direction;

	vector<Vector3R> particlePositions;
	vector<Vector3R> particleVelocities;
	vector<Real> particleDensities;

	particlePositions.resize(totalNumOfPrticles);
	particleDensities.resize(totalNumOfPrticles);
	particleVelocities.resize(totalNumOfPrticles);

	Real posX = lowerCorner[0];

	#pragma omp parallel for schedule(static) firstprivate(posX)

	for(int i = 0; i < num_of_part_x_direction; i++) {

		posX = lowerCorner[0] + i * delX;

		Real posY = lowerCorner[1];

		for(int j = 0; j < num_of_part_y_direction; j++) {

			Real posZ = lowerCorner[2];

			for(int k = 0; k < num_of_part_z_direction; k++, posZ += delZ) {

				size_t index = i * num_of_part_y_direction * num_of_part_z_direction + j * num_of_part_z_direction + k;

				assert(index < totalNumOfPrticles && index >= 0);

				particlePositions[index] = {posX, posY, posZ};
			}
			posY += delY;
		}
	}
	Real width = fabs(distVector[0]) + samplingDistance;
	Real height = fabs(distVector[1]) + samplingDistance;
	Real length = fabs(distVector[2]) + samplingDistance;

	Real fluidVolume = width * height * length;

	return new FluidSystem(particlePositions, particleVelocities, particleDensities, restDensity, fluidVolume, eta);
}


static inline bool is_in_triangle(const Vector3R& corner_a, const Vector3R& corner_b, const Vector3R& corner_c, const Vector3R& point_p)
{
	Vector3R sideAB = corner_b - corner_a;
	Vector3R sideAC = corner_c - corner_a;
	Vector3R sidePA = corner_a - point_p;
	Vector3R sidePB = corner_b - point_p;
	Vector3R sidePC = corner_c - point_p;

	Real area_full = sideAB.cross(sideAC).norm();

	Real area_ab = sidePA.cross(sidePB).norm();
	Real area_bc = sidePB.cross(sidePC).norm();
	Real area_ca = sidePC.cross(sidePA).norm();

	return fabs(area_ab + area_bc + area_ca - area_full) < threshold;
}


static Vector3R get_shift(const Vector3R& vec_s, const Vector3R& vec_t, Real margin)
{
	return  margin * margin * sqrt(2.0 / (vec_s.dot(vec_t) + margin * margin)) * (vec_s + vec_t).normalized();
}


static void expand_triangle(const Vector3R& vertex_a, const Vector3R& vertex_b, const Vector3R& vertex_c, const Real margin, Vector3R& vertex_a_prime, Vector3R& vertex_b_prime, Vector3R& vertex_c_prime)
{
	const Vector3R vec_AB = vertex_b - vertex_a;
	const Vector3R vec_BC = vertex_c - vertex_b;
	const Vector3R vec_CA = vertex_a - vertex_c;

	Vector3R norm_AB = vec_AB.cross(vec_BC).cross(vec_AB);
	Vector3R norm_BC = vec_BC.cross(vec_CA).cross(vec_BC);
	Vector3R norm_CA = vec_CA.cross(vec_AB).cross(vec_CA);

	norm_AB = (norm_AB.dot(vec_BC) * norm_AB).normalized();
	norm_BC = (norm_BC.dot(vec_CA) * norm_BC).normalized();
	norm_CA = (norm_CA.dot(vec_AB) * norm_CA).normalized();

	const Vector3R expansion_AB = - margin * norm_AB;
	const Vector3R expansion_BC = - margin * norm_BC;
	const Vector3R expansion_CA = - margin * norm_CA;

	vertex_a_prime = vertex_a + get_shift(expansion_CA, expansion_AB, margin);
	vertex_b_prime = vertex_b + get_shift(expansion_AB, expansion_BC, margin);
	vertex_c_prime = vertex_c + get_shift(expansion_BC, expansion_CA, margin);
}


void learnSPH::sample_triangle(const Vector3R &vertex_a, const Vector3R &vertex_b, const Vector3R &vertex_c, Real samplingDistance, vector<Vector3R> &borderParticles, bool hexagonal)
{
	vector<Vector3R> faceParticles;

	Vector3R corner_a;
	Vector3R corner_b;
	Vector3R corner_c;

	expand_triangle(
		vertex_a,
		vertex_b,
		vertex_c,
		samplingDistance / 2.0,
		corner_a,
		corner_b,
		corner_c);

	Vector3R vec_AB = corner_b - corner_a;
	Vector3R vec_AC = corner_c - corner_a;

	assert(1 - fabs(vec_AB.normalized().dot(vec_AC.normalized())) > threshold);

	Vector3R vec_BC = corner_c - corner_b;
	Vector3R vec_BA = corner_a - corner_b;

	assert(1 - fabs(vec_BC.normalized().dot(vec_BA.normalized())) > threshold);

	Vector3R vec_CA = corner_a - corner_c;
	Vector3R vec_CB = corner_b - corner_c;

	assert(1 - fabs(vec_CA.normalized().dot(vec_CB.normalized())) > threshold);

	Vector3R vec_major;
	Vector3R vec_normal;
	Vector3R vec_subordinate;
	Vector3R reference;

	if (vec_AB.dot(vec_AC) + threshold < 0)	{
		vec_major = vec_BC;
		vec_subordinate = vec_BA;
		reference = corner_b;
	} else if (vec_BC.dot(vec_BA) + threshold < 0) {
		vec_major = vec_CA;
		vec_subordinate = vec_CB;
		reference = corner_c;
	} else if (vec_CA.dot(vec_CB) + threshold < 0) {
		vec_major = vec_AB;
		vec_subordinate = vec_AC;
		reference = corner_a;
	} else {
		vec_major = vec_AB;
		vec_subordinate = vec_AC;
		reference = corner_a;
	}
	vec_normal = vec_major.cross(vec_subordinate).cross(vec_major).normalized();

	vec_normal = vec_normal.dot(vec_subordinate) * vec_normal;

	if (hexagonal) {

		Real samp_dist_major = samplingDistance * sqrt(3.0) / 2.0;
		Real samp_dist_normal = samplingDistance;

		Vector3R vec_major_unit = vec_major.normalized() * samp_dist_major;
		Vector3R vec_normal_unit = vec_normal.normalized() * samp_dist_normal;

		int maxi = (vec_major.norm() / samp_dist_major) + 1;
		int maxj = (vec_normal.norm() / samp_dist_normal) + 1;

		for(int i = 0; i < maxi; i++) {

			if (i % 2 == 1) {
				Vector3R new_point = reference + i * vec_major_unit + 0.5 * vec_normal_unit;

				while(is_in_triangle(corner_a, corner_b, corner_c, new_point)) {
					
					faceParticles.push_back(new_point);
					new_point = new_point + vec_normal_unit;
				}
			} else {
				Vector3R new_point = reference + i * vec_major_unit;

				while(is_in_triangle(corner_a, corner_b, corner_c, new_point)) {
					
					faceParticles.push_back(new_point);
					new_point = new_point + vec_normal_unit;
				}
			}
		}
	} else {

		Vector3R vec_major_unit = vec_major.normalized() * samplingDistance;

		Vector3R vec_normal_unit = vec_normal.normalized() * samplingDistance;

		int maxi = (vec_major.norm() / samplingDistance) + 1;
		int maxj = (vec_normal.norm() / samplingDistance) + 1;

		for(int i = 0; i < maxi; i++) {
			
			Vector3R new_point = reference + i * vec_major_unit;

			while(is_in_triangle(corner_a, corner_b, corner_c, new_point)) {

				faceParticles.push_back(new_point);
				new_point = new_point + vec_normal_unit;
			}
		}
	} 

	Vector3R centroid = (vertex_a + vertex_b + vertex_c) / 3.0;

	Vector3R mass_center = Vector3R(0.0, 0.0, 0.0);

	for (Vector3R &pt : faceParticles) { mass_center += pt; }

	mass_center /= float(faceParticles.size());

	Vector3R vec_offset = centroid - mass_center;

	for (Vector3R &pt : faceParticles) { pt += vec_offset; }

	borderParticles.insert(borderParticles.end(), faceParticles.begin(), faceParticles.end());
}

void learnSPH::sample_border_model_surface(vector<Vector3R>& borderParticles, const Matrix4d transitionMatr, const string& patToModel, Real samplingDistance)
{
	vector<Vector3R> genBorderParticles;

	FaceVertexList* listOfVerticesAndFaces = parse_wavefront(patToModel);
	if(listOfVerticesAndFaces == nullptr){
		return;
	}

	const auto& vertices = listOfVerticesAndFaces->vecrtexes;
	const auto& faces    = listOfVerticesAndFaces->faceIndexes;
	size_t counter = 0;
	for(const Vector3i& face : faces){
		assert(face[0] < vertices.size() &&
			face[1] < vertices.size() &&
			face[2] < vertices.size());
		Vector4d vertex1 = transitionMatr*Vector4d(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2], 1);
		Vector4d vertex2 = transitionMatr*Vector4d(vertices[face[1]][0], vertices[face[1]][1], vertices[face[1]][2], 1);
		Vector4d vertex3 = transitionMatr*Vector4d(vertices[face[2]][0], vertices[face[2]][1], vertices[face[2]][2], 1);

		sample_triangle(Vector3R(vertex1[0], vertex1[1], vertex1[2]), 
						Vector3R(vertex2[0], vertex2[1], vertex2[2]),
						Vector3R(vertex3[0], vertex3[1], vertex3[2]),
						samplingDistance,
						genBorderParticles,
						true);
		counter++;
		fprintf(stderr, "\33[2K\rgenerated [%d/%d] faces", counter,faces.size());
	}
	delete listOfVerticesAndFaces;
	borderParticles.swap(genBorderParticles);
}

BorderSystem* learnSPH::sample_border_model(const Matrix4d& transitionMatr, const string& patToModel, Real restDensity, Real samplingDistance, Real eta)
{
	fprintf(stderr, "building model from %s\n", patToModel.c_str());
	vector<Vector3R> borderParticles;
	sample_border_model_surface(borderParticles, transitionMatr, patToModel, samplingDistance);
	auto ret =  new BorderSystem(borderParticles, restDensity, samplingDistance, eta);
	fprintf(stderr, "model mesh generation finished\n", patToModel.c_str());
	return ret;

}



BorderSystem* learnSPH::sample_border_box(const Vector3R &lowerCorner, const Vector3R &upperCorner, Real restDensity, Real samplingDistance, Real eta, bool hexagonal)
{
	vector<Vector3R> borderParticles;

	Vector3R vertexA = Vector3R(lowerCorner(0), lowerCorner(1), lowerCorner(2));
	Vector3R vertexB = Vector3R(lowerCorner(0), upperCorner(1), lowerCorner(2));
	Vector3R vertexD = Vector3R(upperCorner(0), lowerCorner(1), lowerCorner(2));
	Vector3R vertexC = Vector3R(upperCorner(0), upperCorner(1), lowerCorner(2));
	Vector3R vertexE = Vector3R(lowerCorner(0), lowerCorner(1), upperCorner(2));
	Vector3R vertexF = Vector3R(lowerCorner(0), upperCorner(1), upperCorner(2));
	Vector3R vertexH = Vector3R(upperCorner(0), lowerCorner(1), upperCorner(2));
	Vector3R vertexG = Vector3R(upperCorner(0), upperCorner(1), upperCorner(2));

	sample_triangle(vertexA, vertexB, vertexC, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexA, vertexD, vertexC, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexD, vertexH, vertexG, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexD, vertexC, vertexG, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexB, vertexC, vertexG, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexB, vertexF, vertexG, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexA, vertexD, vertexH, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexA, vertexE, vertexH, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexE, vertexF, vertexG, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexE, vertexH, vertexG, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexA, vertexE, vertexF, samplingDistance, borderParticles, hexagonal);
	sample_triangle(vertexA, vertexB, vertexF, samplingDistance, borderParticles, hexagonal);

	return new BorderSystem(borderParticles, restDensity, samplingDistance, eta);
}


void learnSPH::sample_ring(vector<Vector3R> &borderParticles, const Vector3R &center, Real radius, const Vector3R &unit_x, const Vector3R &unit_y, Real samplingDistance)
{
	size_t n_samples = ceil(2 * PI * radius / samplingDistance);

	auto unit_rad = 2 * PI / n_samples;

	for(int j = 0; j < n_samples; j++){

		Real x = radius * std::sin(unit_rad * j);

		Real y = radius * std::cos(unit_rad * j);

		Vector3R pt = center + x * unit_x + y * unit_y;

		borderParticles.push_back(pt);
	}
}


BorderSystem* learnSPH::sample_border_cone(Real lowerRadius, const Vector3R &lowerCenter, Real upperRadius, const Vector3R &upperCenter, Real restDensity, Real samplingDistance, Real eta)
{
	vector<Vector3R> borderParticles;

	Vector3R normal = upperCenter - lowerCenter;

	Vector3R axis_alpha = normal.cross(Vector3R(0.0, 0.0, 1.0));

	if(axis_alpha.norm() < threshold) axis_alpha = normal.cross(Vector3R(0.0, 1.0, 0.0));

	assert(axis_alpha.norm() > threshold);

	Vector3R axis_beta = normal.cross(axis_alpha);

	axis_alpha = axis_alpha.normalized();
	axis_beta = axis_beta.normalized();

	size_t circleCnt = sqrt(normal.squaredNorm() + pow2(upperRadius - lowerRadius)) / samplingDistance + 1;

	auto unit_normal = normal / (circleCnt - 1);

	for (int i = 0; i < circleCnt; i++) {

		auto radius = lowerRadius + (upperRadius - lowerRadius) * i / (circleCnt - 1);

		Vector3R center = lowerCenter + i * unit_normal;

		sample_ring(borderParticles, center, radius, axis_alpha, axis_beta, samplingDistance);
	}

	circleCnt = ceil(3 * lowerRadius / samplingDistance / 2);

	for (int i = 1; i < circleCnt; i++) {

		auto radius = lowerRadius * i / circleCnt;

		sample_ring(borderParticles, lowerCenter, radius, axis_alpha, axis_beta, samplingDistance);
	}
	borderParticles.push_back(lowerCenter);

	return new BorderSystem(borderParticles, restDensity, samplingDistance, eta);
}

void learnSPH::sample_sphere(vector<Vector3R>& borderParticles, const Real radius, const Vector3R center, const Real samplingDistance)
{
	auto initialDirection = Vector3R(0.0, 0.0, 1.0);
	auto normalToInitial = Vector3R(0.0, 1.0, 0.0);

	Real alignedFirstSamplingAngle = 2 * PI / int(2 * PI / (samplingDistance / radius));

	Real curFirstSamplingAngle = 0.0;

	while(curFirstSamplingAngle < PI) {

		Real curCircleRadius = radius * sin(curFirstSamplingAngle);

		Real alignedSecondSamplingAngle = 2 * PI / int(2 * PI / (samplingDistance / (curCircleRadius + threshold)));

		Real curSecondAmplingAngle = 0;

		Matrix3d zDirRotate;

		zDirRotate = AngleAxisd(curFirstSamplingAngle, normalToInitial);

		while(curSecondAmplingAngle < 2 * PI) {

			Matrix3d xyDirRotate;

			xyDirRotate = AngleAxisd(curSecondAmplingAngle, initialDirection);

			Vector3R pt = center + radius * xyDirRotate * zDirRotate * initialDirection;

			borderParticles.push_back(pt);

			curSecondAmplingAngle += alignedSecondSamplingAngle;
		}
		curFirstSamplingAngle += alignedFirstSamplingAngle;
	}
}
