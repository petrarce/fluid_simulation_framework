#pragma once
#include <types.hpp>
#include <Eigen/Dense>
#include <vector>
#include <CompactNSearch>
#include <kernel.h>

using namespace std;
using namespace Eigen;
using namespace CompactNSearch;

namespace learnSPH
{
	class ParticleDataSet {
	protected:
		vector<Vector3R> particlePositions;
		Real restDensity;
		Real particleDiameter;
		Real particleMass;

	public:
		enum ParticleType{
			BORDER = 0,
			NORMAL
		};
	public:

		/*
			get type of the particles, that are handled by the set
			currently there are 2 types of particles that are possible: 
				1. Fluid particles (NORMAL)
				2. Border particles (BORDER)
		*/
		virtual ParticleType getType() const
		{
			return NORMAL;
		}

		Real getRestDensity() const
		{
			return this->restDensity;
		}
		Real getParticleDiameter() const
		{
			return this->particleDiameter;
		}

		Real getParticleMass() const
		{
			return this->particleMass;
		}

		vector<Vector3R>& getParticlePositions()
		{
			return particlePositions;
		};

		size_t getNumberOfParticles() const
		{
			return particlePositions.size();
		};

		ParticleDataSet(vector<Vector3R>& particlePositions,
						Real restDensity,
						Real fluidVolume):
			restDensity(restDensity)

		{
			assert(restDensity > 0);
			assert(fluidVolume > 0);
			this->particlePositions.swap(particlePositions);
			this->particleMass = (this->restDensity * fluidVolume)/this->particlePositions.size();
			this->particleDiameter = cbrt(this->particleMass/this->restDensity);
		};
		virtual ~ParticleDataSet(){};
	};

	class BorderPartDataSet :public ParticleDataSet{
	private:
		/*
			vector which contains volume for each Border particle. 
			Border particle algorithm adjusts each particle volume w.r.t. the grid...
			TODO: make clearer explanation...
		*/
		vector<Real> particleVolume;
	public:

		virtual ParticleType getType() const
		{
			return BORDER;
		};

		vector<Real>& getParticleVolume()
		{
			return particleVolume;
		};

		BorderPartDataSet(vector<Vector3R>& particlePositions,
							Real restDensity,
							Real fluidVolume):
			ParticleDataSet(particlePositions, restDensity, fluidVolume)
		{
			this->particleVolume.resize(this->particlePositions.size());
			NeighborhoodSearch ns(this->particleDiameter*1.2, false);
			unsigned int pset = ns.add_point_set(&this->particlePositions[0](0), 
								this->particlePositions.size(),
								true,
								true);
			ns.update_point_sets();
			vector<vector<unsigned int>> neighbours;
			for(int i = 0; i < this->particlePositions.size(); i++){
				ns.find_neighbors(pset, i, neighbours);
				assert(neighbours.size() == 1);
				Real kerel_sum = 0;
				for(int j : neighbours[0]){
					kerel_sum += kernel::kernelFunction(this->particlePositions[i], 
														this->particlePositions[j], 
														this->particleDiameter*1.2*0.5);
				}
				this->particleVolume[i] = (kerel_sum > 0.0) ? (1 / kerel_sum) : learnSPH::kernel::pow3(this->particleDiameter);
			}
		};
		
		~BorderPartDataSet(){};
	};

	class NormalPartDataSet :public ParticleDataSet{
	private:
		vector<Vector3R> particleVelocities;
		vector<Real> particleDencities;
		vector<Vector3R> particleExternalForces;
		Real compactSupportFactor;

	public:

		virtual ParticleType getType() const
		{
			return NORMAL;
		};

		opcode setCompactSupportFactor(const Real val)
		{
			this->compactSupportFactor = val;
			return STATUS_OK;
		}

		opcode setParticlePositions(vector<Vector3R>& newPositions)
		{
			assert(this->particlePositions.size() == newPositions.size());
			this->particlePositions.swap(newPositions);
			return STATUS_OK;
		}

		Real getSmoothingLength()
		{
			return 0.5*this->particleDiameter*this->compactSupportFactor;
		}

		Real getCompactSupport()
		{
			return this->particleDiameter*this->compactSupportFactor;
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
						Real fluidVolume):
						ParticleDataSet(particlePositions, restDensity, fluidVolume), compactSupportFactor(1.2)
		{
			this->particleDencities.swap(particleDencities);
			this->particleVelocities.swap(particleVelocities);
			this->particleExternalForces = vector<Vector3R>(this->particlePositions.size(),Eigen::Vector3d(0.0,0.0,0.0));
		};

        NormalPartDataSet(
						vector<Vector3R>& particlePositions,
						vector<Vector3R>& particleVelocities,
						vector<Real>& particleDencities,
						vector<Vector3R>& particleExternalForces,
						Real restDensity,
						Real fluidVolume):
						ParticleDataSet(particlePositions, restDensity, fluidVolume)
		{
			this->particleDencities.swap(particleDencities);
			this->particleVelocities.swap(particleVelocities);
			this->particleExternalForces.swap(particleExternalForces);
		};

		~NormalPartDataSet(){};
	};
};