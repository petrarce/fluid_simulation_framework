#pragma once
#include <types.hpp>
#include <Eigen/Dense>
#include <vector>
#include <CompactNSearch>
#include <kernel.h>

using namespace std;
using namespace Eigen;
using namespace CompactNSearch;

typedef Vector3R VelocVector;
typedef Vector3R PositionVector;

namespace learnSPH
{
	class ParticleDataSet {
	protected:
		vector<PositionVector> particlePositions;
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
		virtual ParticleType getSetType() const = 0;
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
		
		/*
			get particlePositions vector directly (required for vtk generation)
		*/
		vector<PositionVector>& getParticlePositions() 
		{
			return particlePositions;
		};
		
		const PositionVector* getParticlePositionsData() const
		{
			return static_cast<const PositionVector*>(particlePositions.data());
		}; 
		
		size_t getNumberOfParticles() const
		{
			return particlePositions.size();
		};
		
		ParticleDataSet(vector<PositionVector>& particlePositions, 
						Real restDensity,
						Real particleDiameter):
			restDensity(restDensity),
			particleDiameter(particleDiameter)

		{
			assert(restDensity > 0);
			assert(particleDiameter > 0);
			this->particlePositions.swap(particlePositions);
			this->particleMass = this->restDensity * 
				this->particleDiameter * this->particleDiameter * this->particleDiameter;
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

		virtual ParticleType getSetType() const
		{
			return BORDER;
		};
		
		vector<Real>& getParticleVolume()
		{
			return particleVolume;
		};
		
		const Real* getParticleVolumeData() const
		{
			return static_cast<const Real*>(particleVolume.data());
		};

		BorderPartDataSet(vector<PositionVector>& particlePositions, 
							Real restDensity,
							Real particleDiameter):
			ParticleDataSet(particlePositions, restDensity, particleDiameter)
		{
			this->particleVolume.resize(this->particlePositions.size());
			NeighborhoodSearch ns(1.2*this->particleDiameter, false);
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
														1.2*this->particleDiameter);
				}
				this->particleVolume[i] = 1 / kerel_sum;
			}
		};
		
		~BorderPartDataSet(){};
	};

	class NormalPartDataSet :public ParticleDataSet{
		vector<Vector3R> particleVelocities;
		vector<Real> particleDencities;

	public:

		virtual ParticleType getSetType() const
		{
			return NORMAL;
		};
		
		vector<Real>& getParticleDencities()
		{
			return particleDencities;
		};
		
		const Real* getParticleDencitiesData() const
		{
			return particleDencities.data();
		};
		vector<VelocVector>& getParticleVelocities()
		{
			return particleVelocities;
		};
		
		const VelocVector* getParticleVelocitiesData() const
		{
			return particleVelocities.data();
		};


		NormalPartDataSet(vector<PositionVector>& particlePositions,
							vector<VelocVector>& particleVelocities,
							vector<Real>& particleDencities, 
							Real restDensity,
							Real particleDiameter):

			ParticleDataSet(particlePositions, restDensity, particleDiameter)
		{
			this->particleDencities.swap(particleDencities);
			this->particleVelocities.swap(particleVelocities);
		};

		~NormalPartDataSet(){};
	};
};