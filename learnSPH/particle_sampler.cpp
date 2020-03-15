#include <storage.h>
#include <random>
#include <particle_sampler.h>
#include <kernel.h>
#include <types.hpp>
#include <Eigen/Geometry>

using namespace std;
using namespace learnSPH;
using namespace learnSPH::kernel;


Real learnSPH::sample_cube(vector<Vector3R> &positions, const Vector3R &lowerCorner, const Vector3R &upperCorner, Real samp_dist)
{
	Vector3R distVector = upperCorner - lowerCorner;

	size_t n_div_x = fabs(distVector[0] / samp_dist) + 1;
	size_t n_div_y = fabs(distVector[1] / samp_dist) + 1;
	size_t n_div_z = fabs(distVector[2] / samp_dist) + 1;

	Real delX = samp_dist * distVector[0] / fabs(distVector[0]);
	Real delY = samp_dist * distVector[1] / fabs(distVector[1]);
	Real delZ = samp_dist * distVector[2] / fabs(distVector[2]);

	size_t n_particles = n_div_x * n_div_y * n_div_z;

	default_random_engine engine;

	normal_distribution<double> gauss(0.0, 1.0);

	for(int i = 0; i < n_div_x; i++) {

		Real posX = lowerCorner[0] + i * delX;

		for(int j = 0; j < n_div_y; j++) {

			Real posY = lowerCorner[1] + j * delY;

			for(int k = 0; k < n_div_z; k++) {

				Real posZ = lowerCorner[2] + k * delZ;

				size_t index = i * n_div_y * n_div_z + j * n_div_z + k;

				assert(index < n_particles && index >= 0);

				positions.push_back(Vector3R(posX, posY, posZ) + Vector3R(gauss(engine) * delX / 10.0, gauss(engine) * delY / 10.0, gauss(engine) * delZ / 10.0));
			}
		}
	}
	auto side_x = fabs(distVector[0]) + samp_dist;
	auto side_y = fabs(distVector[1]) + samp_dist;
	auto side_z = fabs(distVector[2]) + samp_dist;

	return side_x * side_y * side_z;
}


FluidSystem* learnSPH::single_dam(const Vector3R &lowerA, const Vector3R &upperA, Real restDensity, Real samp_dist, Real eta)
{
	vector<Vector3R> positions;
	vector<Vector3R> velocities;
	vector<Real> densities;

	Real volumeA = sample_cube(positions, lowerA, upperA, samp_dist);

	auto n_particles = positions.size();

	velocities.assign(n_particles, Vector3R(0.0, 0.0, 0.0));
	densities.resize(n_particles, 0.0);

	return new FluidSystem(positions, velocities, densities, restDensity, volumeA, eta);
}


FluidSystem* learnSPH::double_dam(const Vector3R &lowerA, const Vector3R &upperA, const Vector3R &lowerB, const Vector3R &upperB, Real restDensity, Real samp_dist, Real eta)
{
	vector<Vector3R> positions;
	vector<Vector3R> velocities;
	vector<Real> densities;

	Real volumeA = sample_cube(positions, lowerA, upperA, samp_dist);
	Real volumeB = sample_cube(positions, lowerB, upperB, samp_dist);

	auto n_particles = positions.size();

	velocities.assign(n_particles, Vector3R(0.0, 0.0, 0.0));
	densities.assign(n_particles, 0.0);

	return new FluidSystem(positions, velocities, densities, restDensity, volumeA + volumeB, eta);
}


void learnSPH::sample_shell(vector<Vector3R> &positions, const Vector3R &center, Real radius, Real samp_dist)
{
	auto unit_x_axis = Vector3R(0.0, 0.0, 1.0);
	auto unit_z_axis = Vector3R(0.0, 1.0, 0.0);

	Real alignedFirstSamplingAngle = 2 * PI / int(2 * PI * radius / samp_dist);

	Real curFirstSamplingAngle = 0.0;

	while(curFirstSamplingAngle < PI) {

		Real curCircleRadius = radius * sin(curFirstSamplingAngle);

		Real alignedSecondSamplingAngle = 2 * PI / int(2 * PI * curCircleRadius / samp_dist);

		Real curSecondAmplingAngle = 0;

		Matrix3d zDirRotate;

		zDirRotate = AngleAxisd(curFirstSamplingAngle, unit_z_axis);

		while(curSecondAmplingAngle < 2 * PI) {

			Matrix3d xyDirRotate;

			xyDirRotate = AngleAxisd(curSecondAmplingAngle, unit_x_axis);

			Vector3R pt = center + radius * xyDirRotate * zDirRotate * unit_x_axis;

			positions.push_back(pt);

			curSecondAmplingAngle += alignedSecondSamplingAngle;
		}
		curFirstSamplingAngle += alignedFirstSamplingAngle;
	}
}


FluidSystem* learnSPH::water_drop(const Vector3R &center, const Vector3R &speed, Real radius, Real restDensity, Real samp_dist, Real eta)
{
	vector<Vector3R> positions;
	vector<Vector3R> velocities;
	vector<Real> densities;

	int n_shells = ceil(radius / samp_dist);

	for (int i_shell = 0; i_shell < n_shells; i_shell++) {

		auto curr_radius = radius * (i_shell + 1) / n_shells;

		sample_shell(positions, center, curr_radius, samp_dist);
	}
	velocities.assign(positions.size(), speed);
	densities.assign(positions.size(), 0.0);

	auto volume = 4.0 * PI * pow3(radius) / 3.0;

	return new FluidSystem(positions, velocities, densities, restDensity, volume, eta);
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

	return fabs(area_ab + area_bc + area_ca - area_full) < 1e-6;
}


static Vector3R get_shift(const Vector3R &vec_s, const Vector3R &vec_t, Real margin)
{
	return margin * sqrt(2.0 / (vec_s.dot(vec_t) + 1.0)) * (vec_s + vec_t).normalized();
}


static void expand_triangle(const Vector3R& vertex_a, const Vector3R& vertex_b, const Vector3R& vertex_c, const Real margin, Vector3R& vertex_a_prime, Vector3R& vertex_b_prime, Vector3R& vertex_c_prime)
{
	const Vector3R vec_ab = vertex_b - vertex_a;
	const Vector3R vec_BC = vertex_c - vertex_b;
	const Vector3R vec_CA = vertex_a - vertex_c;

	Vector3R norm_AB = vec_ab.cross(vec_BC).cross(vec_ab);
	Vector3R norm_BC = vec_BC.cross(vec_CA).cross(vec_BC);
	Vector3R norm_CA = vec_CA.cross(vec_ab).cross(vec_CA);

	norm_AB = - norm_AB.normalized();
	norm_BC = - norm_BC.normalized();
	norm_CA = - norm_CA.normalized();

	vertex_a_prime = vertex_a + get_shift(norm_CA, norm_AB, margin);
	vertex_b_prime = vertex_b + get_shift(norm_AB, norm_BC, margin);
	vertex_c_prime = vertex_c + get_shift(norm_BC, norm_CA, margin);
}


void learnSPH::sample_triangle(const Vector3R &vertex_a, const Vector3R &vertex_b, const Vector3R &vertex_c, Real samp_dist, vector<Vector3R> &particles)
{
	vector<Vector3R> positions;

	Vector3R corner_a;
	Vector3R corner_b;
	Vector3R corner_c;

	expand_triangle(vertex_a, vertex_b, vertex_c, samp_dist / 2.0, corner_a, corner_b, corner_c);

	Vector3R vec_ab = corner_b - corner_a;
	Vector3R vec_ac = corner_c - corner_a;

	Vector3R vec_hb = vec_ab - vec_ab.dot(vec_ac.normalized()) * vec_ac.normalized();

	Vector3R unit_ac = vec_ac.normalized() * cos(PI / 6) * samp_dist;
	Vector3R unit_hb = vec_hb.normalized() * sin(PI / 6) * samp_dist;

	Vector3R cursor = corner_a;

	while ((cursor - corner_a).norm() <= vec_ac.norm()) {

		Vector3R bicursor = cursor;

		while((bicursor - cursor).norm() <= vec_hb.norm()) {

			if (!is_in_triangle(corner_a, corner_b, corner_c, bicursor)) break;

			positions.push_back(bicursor);

			bicursor += unit_hb * 2;
		}
		cursor += unit_ac;

		bicursor = cursor + unit_hb;

		while((bicursor - cursor).norm() <= vec_hb.norm()) {

			if (!is_in_triangle(corner_a, corner_b, corner_c, bicursor)) break;

			positions.push_back(bicursor);

			bicursor += unit_hb * 2;
		}
		cursor += unit_ac;
	}
	Vector3R centroid = (vertex_a + vertex_b + vertex_c) / 3.0;

	Vector3R mass_center = Vector3R(0.0, 0.0, 0.0);

	for (Vector3R &pt : positions) { mass_center += pt; }

	mass_center /= float(positions.size());

	Vector3R vec_offset = centroid - mass_center;

	for (Vector3R &pt : positions) { pt += vec_offset; }

	particles.insert(particles.end(), positions.begin(), positions.end());
}


BorderSystem* learnSPH::sample_border_box(const Vector3R &lowerCorner, const Vector3R &upperCorner, Real restDensity, Real samp_dist, Real eta)
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

	sample_triangle(vertexA, vertexD, vertexC, samp_dist, borderParticles);
	sample_triangle(vertexA, vertexB, vertexC, samp_dist, borderParticles);
	sample_triangle(vertexD, vertexH, vertexG, samp_dist, borderParticles);
	sample_triangle(vertexD, vertexC, vertexG, samp_dist, borderParticles);
	sample_triangle(vertexB, vertexC, vertexG, samp_dist, borderParticles);
	sample_triangle(vertexB, vertexF, vertexG, samp_dist, borderParticles);
	sample_triangle(vertexA, vertexD, vertexH, samp_dist, borderParticles);
	sample_triangle(vertexA, vertexE, vertexH, samp_dist, borderParticles);
	sample_triangle(vertexE, vertexF, vertexG, samp_dist, borderParticles);
	sample_triangle(vertexE, vertexH, vertexG, samp_dist, borderParticles);
	sample_triangle(vertexA, vertexE, vertexF, samp_dist, borderParticles);
	sample_triangle(vertexA, vertexB, vertexF, samp_dist, borderParticles);

	return new BorderSystem(borderParticles, restDensity, samp_dist, eta);
}


void learnSPH::sample_ring(vector<Vector3R> &borderParticles, const Vector3R &center, Real radius, const Vector3R &unit_x, const Vector3R &unit_y, Real samp_dist)
{
	size_t n_samples = ceil(2 * PI * radius / samp_dist);

	auto unit_rad = 2 * PI / n_samples;

	for(int j = 0; j < n_samples; j++){

		Real x = radius * std::sin(unit_rad * j);

		Real y = radius * std::cos(unit_rad * j);

		Vector3R pt = center + x * unit_x + y * unit_y;

		borderParticles.push_back(pt);
	}
}


BorderSystem* learnSPH::sample_border_cone(Real lowerRadius, const Vector3R &lowerCenter, Real upperRadius, const Vector3R &upperCenter, Real restDensity, Real samp_dist, Real eta)
{
	vector<Vector3R> borderParticles;

	Vector3R normal = upperCenter - lowerCenter;

	Vector3R axis_alpha = normal.cross(Vector3R(0.0, 0.0, 1.0));

	if(axis_alpha.norm() < 1e-6) axis_alpha = normal.cross(Vector3R(0.0, 1.0, 0.0));

	assert(axis_alpha.norm() > 1e-6);

	Vector3R axis_beta = normal.cross(axis_alpha);

	axis_alpha = axis_alpha.normalized();
	axis_beta = axis_beta.normalized();

	size_t circleCnt = sqrt(normal.squaredNorm() + pow2(upperRadius - lowerRadius)) / samp_dist + 1;

	auto unit_normal = normal / (circleCnt - 1);

	for (int i = 0; i < circleCnt; i++) {

		auto radius = lowerRadius + (upperRadius - lowerRadius) * i / (circleCnt - 1);

		Vector3R center = lowerCenter + i * unit_normal;

		sample_ring(borderParticles, center, radius, axis_alpha, axis_beta, samp_dist);
	}
	return new BorderSystem(borderParticles, restDensity, samp_dist, eta);
}


BorderSystem* learnSPH::sample_border_sphere(Real radius, const Vector3R &center, Real restDensity, Real samp_dist, Real eta)
{
	vector<Vector3R> borderParticles;

	auto unit_x_axis = Vector3R(0.0, 0.0, 1.0);
	auto unit_z_axis = Vector3R(0.0, 1.0, 0.0);

	Real alignedFirstSamplingAngle = 2 * PI / int(2 * PI * radius / samp_dist);

	Real curFirstSamplingAngle = 0.0;

	while(curFirstSamplingAngle < PI) {

		Real curCircleRadius = radius * sin(curFirstSamplingAngle);

		Real alignedSecondSamplingAngle = 2 * PI / int(2 * PI * curCircleRadius / samp_dist);

		Real curSecondAmplingAngle = 0;

		Matrix3d zDirRotate;

		zDirRotate = AngleAxisd(curFirstSamplingAngle, unit_z_axis);

		while(curSecondAmplingAngle < 2 * PI) {

			Matrix3d xyDirRotate;

			xyDirRotate = AngleAxisd(curSecondAmplingAngle, unit_x_axis);

			Vector3R pt = center + radius * xyDirRotate * zDirRotate * unit_x_axis;

			borderParticles.push_back(pt);

			curSecondAmplingAngle += alignedSecondSamplingAngle;
		}
		curFirstSamplingAngle += alignedFirstSamplingAngle;
	}
	return new BorderSystem(borderParticles, restDensity, samp_dist, eta);
}
