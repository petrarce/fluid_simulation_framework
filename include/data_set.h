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
		virtual ParticleType getSetType() = 0;

		Real getParticleDiameter()
		{
			return this->particleDiameter;
		}

		Real getParticleMass()
		{
			return this->particleMass;
		}
		
		/*
			get particlePositions vector directly (required for vtk geeration)
		*/
		const vector<PositionVector>& getParticlePositions()
		{
			return particlePositions;
		};
		
		PositionVector* getParticlePositionsData()
		{
			return particlePositions.data();
		};
		
		size_t getNumberOfParticles()
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
			Border particle alhorythm adjusts each particle volume w.r.t. the grid...
			TODO: make clearer explenation...
		*/
		vector<Real> particleVolume;
	public:

		virtual ParticleType getSetType()
		{
			return BORDER;
		};
		
		const vector<Real>& getParticleVolume()
		{
			return particleVolume;
		};
		
		Real* getParticleVolumeData()
		{
			return particleVolume.data();
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

		virtual ParticleType getSetType()
		{
			return NORMAL;
		};
		
		const vector<Real>& getParticleDencities()
		{
			return particleDencities;
		};
		
		Real* getParticleDencitiesData()
		{
			return particleDencities.data();
		};
		const vector<VelocVector>& getParticleVelocities()
		{
			return particleVelocities;
		};
		
		VelocVector* getParticleVelocitiesData()
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