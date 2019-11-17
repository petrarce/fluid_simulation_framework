#pragma once
#include <types.hpp>
#include <Eigen/Dense>
#include <vector>
#include <CompactNSearch>
#include <kernel.h>

using namespace std;
using namespace Eigen;
using namespace CompactNSearch;
using namespace learnSPH::kernel;

namespace learnSPH
{
	class ParticleSystem
	{
		protected:
			vector<Vector3R> positions;
			Real restDensity;

		public:
			Real getRestDensity() const
			{
				return this->restDensity;
			}

			vector<Vector3R>& getPositions()
			{
				return positions;
			};

			size_t size() const
			{
				return positions.size();
			};

			ParticleSystem(vector<Vector3R>& positions, Real restDensity):restDensity(restDensity)
			{
				assert(restDensity > 0);

				this->positions.assign(positions.begin(), positions.end());
			};

			virtual ~ParticleSystem(){};
	};

	class BorderSystem:public ParticleSystem
	{
		private:
			vector<Real> volumes;

		public:
			vector<Real>& getVolumes()
			{
				return volumes;
			};

			BorderSystem(vector<Vector3R>& positions, Real restDensity, Real diameter, Real eta):ParticleSystem(positions, restDensity)
			{
				this->volumes.resize(this->positions.size());

				NeighborhoodSearch ns(diameter * eta * 2.0, false);

				ns.add_point_set((Real*)(this->positions.data()), this->positions.size(), false);

				ns.update_point_sets();

				vector<vector<unsigned int> > neighbours;

				for(int i = 0; i < this->positions.size(); i++){

					ns.find_neighbors(0, i, neighbours);

					if (neighbours[0].empty()) {

						this->volumes[i] = pow3(diameter);
						continue;
					}
					Real sum = 0.0;

					for(int j : neighbours[0]) sum += kernelFunction(this->positions[i], this->positions[j], diameter * eta);

					assert(sum > 0.0);

					this->volumes[i] = 1.0 / sum;
				}
			};

			~BorderSystem(){};
	};

	class FluidSystem:public ParticleSystem
	{
		private:
			Real mass;
			Real diameter;
			Real smooth_length;
			Real compact_support;

			vector<Real> dencities;
			vector<Vector3R> velocities;
			vector<Vector3R> external_forces;

		public:
			void setPositions(vector<Vector3R>& newPositions)
			{
				assert(this->positions.size() == newPositions.size());
				this->positions.assign(newPositions.begin(), newPositions.end());
			}

			Real getDiameter() const
			{
				return this->diameter;
			}

			Real getMass() const
			{
				return this->mass;
			}

			Real getSmoothingLength()
			{
				return this->smooth_length;
			}

			Real getCompactSupport()
			{
				return this->compact_support;
			}

			vector<Real>& getDensities()
			{
				return dencities;
			};

			vector<Vector3R>& getVelocities()
			{
				return velocities;
			};

			vector<Vector3R>& getExternalForces()
			{
				return external_forces;
			};

			FluidSystem(
							vector<Vector3R>& positions,
							vector<Vector3R>& velocities,
							vector<Real>& dencities,
							Real restDensity,
							Real fluidVolume,
							Real eta):ParticleSystem(positions, restDensity)
			{
				this->mass = (this->restDensity * fluidVolume) / this->positions.size();
				this->diameter = cbrt(this->mass / this->restDensity);
				this->smooth_length = eta * this->diameter;
				this->compact_support = 2.0 * this->smooth_length;

				this->dencities.assign(dencities.begin(), dencities.end());
				this->velocities.assign(velocities.begin(), velocities.end());
				this->external_forces = vector<Vector3R>(this->positions.size(), Vector3R(0.0, 0.0, 0.0));
			};
			~FluidSystem(){};
	};
};
