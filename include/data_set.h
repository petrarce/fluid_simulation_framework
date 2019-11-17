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
	class ParticleDataSet
	{
		protected:
			vector<Vector3R> particlePositions;
			Real restDensity;

		public:
			Real getRestDensity() const
			{
				return this->restDensity;
			}

			vector<Vector3R>& getParticlePositions()
			{
				return particlePositions;
			};

			size_t size() const
			{
				return particlePositions.size();
			};

			ParticleDataSet(vector<Vector3R>& particlePositions, Real restDensity):restDensity(restDensity)
			{
				assert(restDensity > 0);

				this->particlePositions.assign(particlePositions.begin(), particlePositions.end());
			};

			virtual ~ParticleDataSet(){};
	};

	class BorderPartDataSet:public ParticleDataSet
	{
		private:
			vector<Real> particleVolumes;

		public:
			vector<Real>& getParticleVolumes()
			{
				return particleVolumes;
			};

			BorderPartDataSet(vector<Vector3R>& particlePositions, Real restDensity, Real diameter, Real eta):ParticleDataSet(particlePositions, restDensity)
			{
				this->particleVolumes.resize(this->particlePositions.size());

				NeighborhoodSearch ns(diameter * eta * 2.0, false);

				ns.add_point_set((Real*)(this->particlePositions.data()), this->particlePositions.size(), false);

				ns.update_point_sets();

				vector<vector<unsigned int> > neighbours;

				for(int i = 0; i < this->particlePositions.size(); i++){

					ns.find_neighbors(0, i, neighbours);

					if (neighbours[0].empty()) {

						this->particleVolumes[i] = pow3(diameter);
						continue;
					}
					Real sum = 0.0;

					for(int j : neighbours[0]) sum += kernelFunction(this->particlePositions[i], this->particlePositions[j], diameter * eta);

					assert(sum > 0.0);

					this->particleVolumes[i] = 1.0 / sum;
				}
			};

			~BorderPartDataSet(){};
	};

	class NormalPartDataSet:public ParticleDataSet
	{
		private:
			Real particleMass;
			Real particleDiameter;
			Real smoothingLength;
			Real compactSupport;

			vector<Real> particleDencities;
			vector<Vector3R> particleVelocities;
			vector<Vector3R> particleExternalForces;

		public:
			void setParticlePositions(vector<Vector3R>& newPositions)
			{
				assert(this->particlePositions.size() == newPositions.size());
				this->particlePositions.assign(newPositions.begin(), newPositions.end());
			}

			Real getParticleDiameter() const
			{
				return this->particleDiameter;
			}

			Real getParticleMass() const
			{
				return this->particleMass;
			}

			Real getSmoothingLength()
			{
				return this->smoothingLength;
			}

			Real getCompactSupport()
			{
				return this->compactSupport;
			}

			vector<Real>& getParticleDencities()
			{
				return particleDencities;
			};

			vector<Vector3R>& getParticleVelocities()
			{
				return particleVelocities;
			};

			vector<Vector3R>& getExternalForces()
			{
				return particleExternalForces;
			};

			NormalPartDataSet(
							vector<Vector3R>& particlePositions,
							vector<Vector3R>& particleVelocities,
							vector<Real>& particleDencities,
							Real restDensity,
							Real fluidVolume,
							Real eta):ParticleDataSet(particlePositions, restDensity)
			{
				this->particleMass = (this->restDensity * fluidVolume) / this->particlePositions.size();
				this->particleDiameter = cbrt(this->particleMass / this->restDensity);
				this->smoothingLength = eta * this->particleDiameter;
				this->compactSupport = 2.0 * this->smoothingLength;

				this->particleDencities.assign(particleDencities.begin(), particleDencities.end());
				this->particleVelocities.assign(particleVelocities.begin(), particleVelocities.end());
				this->particleExternalForces = vector<Vector3R>(this->particlePositions.size(), Vector3R(0.0, 0.0, 0.0));
			};
			~NormalPartDataSet(){};
	};
};
