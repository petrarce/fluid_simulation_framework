#pragma once
#include <types.hpp>
#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

typedef Vector3R VelocVector;
typedef Vector3R PositionVector;

namespace learnSPH
{
	class ParticleDataSet {
	protected:
		vector<PositionVector> particlePositions;
		Real restDensity;
		Real particleDiameter;

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
						Real restDensity):
			restDensity(restDensity)
		{
			this->particlePositions.swap(particlePositions);
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
							Real restDensity):
			ParticleDataSet(particlePositions, restDensity){};
		
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
							Real restDensity):
			ParticleDataSet(particlePositions, restDensity)
		{
			this->particleDencities.swap(particleDencities);
			this->particleVelocities.swap(particleVelocities);
		};
		~NormalPartDataSet(){};
	};
};