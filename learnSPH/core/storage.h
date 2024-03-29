#pragma once
#include <types.hpp>
#include <Eigen/Dense>
#include <vector>
#include <CompactNSearch>
#include <learnSPH/core/kernel.h>
#include <set>
#include <list>

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

			const vector<Vector3R>& getPositions() const
			{
				return positions;
			};
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
				//use swap, because it is constant time operation
				this->positions.swap(positions);
			};

			ParticleSystem(Real restDensity):restDensity(restDensity){};
			ParticleSystem(){};
			virtual ~ParticleSystem(){};
	};

	class BorderSystem:public ParticleSystem
	{
		private:
			vector<Real> volumes;

		public:
			const vector<Real>& getVolumes() const
			{
				return volumes;
			};

			vector<Real>& getVolumes()
			{
				return volumes;
			};
			BorderSystem(){};
			BorderSystem(vector<Vector3R>& positions, Real restDensity, Real diameter, Real eta):ParticleSystem(positions, restDensity)
			{
				this->volumes.resize(this->positions.size());

				Real compactSupport = diameter * eta;
				NeighborhoodSearch ns(compactSupport, false);

				ns.add_point_set((Real*)(this->positions.data()), this->positions.size(), false);

				ns.update_point_sets();

				set<unsigned int> deleted;
				//foreach point find neighbors
				fprintf(stderr, "\n");
				for(size_t i = 0; i < this->size(); i++){
					if(deleted.find(i) != deleted.end()){
						continue;
					}
					vector<vector<unsigned int>> neighbours;
					ns.find_neighbors(0, i, neighbours);
					//find all neighboring points, that are within a threshold from current
					for(int j : neighbours[0]){
						//put all such points into deleted list
						if((this->positions[i] - this->positions[j]).norm() < 0.1 * compactSupport){
							deleted.insert(j);
						}
					}

					//compute volume for of boundary particle
					Real sum = 0.0;
					for(int j : neighbours[0]){
						sum += kernelFunction(this->positions[i], 
												this->positions[j], 
												0.5 * compactSupport);
					}
					if(sum < threshold){
						this->volumes[i] = pow3(diameter);
						continue;
					}
					this->volumes[i] = 1.0 / sum;
					if(i%100000){
						fprintf(stderr, "\33[2K\rprocessing [%lu/%lu] border particles.", i, this->size());
					}

				};

				//remove all points, that are in deleted list
				printf("removing %lu out of %lu border particles. ", deleted.size(), this->size());
				for(auto i = deleted.rbegin(); i != deleted.rend(); i++){
					unsigned int ind = *i;
					this->positions[ind] = this->positions.back();
					this->positions.pop_back();
					this->volumes[ind] = this->volumes.back();
					this->volumes.pop_back();
				}
			};

			~BorderSystem(){};
	};

	class FluidSystem:public ParticleSystem
	{
		private:
			typedef struct Emiter_t{
				Vector3R pos;
				list<size_t> chunkOffsets;
				size_t chunkSize;
				size_t maxChunks;
				size_t chunksCnt;
				Real prevEmmitionTime;
				Vector3R emitVelocity;
				Real emiterArea;
			} Emiter;

			Real mass;
			Real diameter;
			Real mEta {1};

			vector<Real> densities;
			vector<Vector3R> velocities;
			vector<Vector3R> external_forces;

			vector<vector<vector<unsigned int> > > neighbors;

			vector<Emiter> emiters;

			void sample_emiter_fluid_particles(vector<Vector3R>& emitedParticlePositions, 
															const Vector3R& velocityVector, 
															const Real samplingDistance, 
															const Vector3R& emiterPosition)
			{
				assert(samplingDistance > 0);
				assert(velocityVector.norm() > threshold);
				const size_t partCnt = emitedParticlePositions.size();
				const auto& epos = emiterPosition;
				const auto& norm = velocityVector.normalized();
				//perorm Gram-Schmidt process
				Vector3R tg1 = Vector3R(0,0,1);
				if(1 - fabs(tg1.dot(norm)) < threshold){
					tg1 = Vector3R(0,1,0);
				}
				Vector3R tg2 = tg1.cross(norm).normalized();
				tg1 = tg2.cross(norm).normalized();
				Vector3R initPos =  epos - 0.5 * (tg1 + tg2) * sqrt(partCnt) * samplingDistance;
				for(size_t i = 0; i < partCnt; i++){
					unsigned int j = i / sqrt(partCnt);
					unsigned int k = i % int(sqrt(partCnt));
					emitedParticlePositions[i] = initPos + (tg1 * j + tg2 * k) * samplingDistance;
				}
			}

		public:

			using EmiterId = size_t;
			size_t emiters_size()
			{
				return emiters.size();
			}
			EmiterId add_emitter(size_t maxNumOfParticles, 
									Vector3R emmiterPosition, 
									Real emiterArea, 
									Vector3R emmitionVelocity)
			{
				assert(emiterArea > 0);
				assert(emmitionVelocity.norm() > threshold);
				Emiter em;
				em.pos = emmiterPosition;
				em.chunkSize = emiterArea / pow2(this->diameter);
				em.maxChunks = maxNumOfParticles / em.chunkSize;
				em.prevEmmitionTime = 0;
				em.emitVelocity = emmitionVelocity;
				em.emiterArea = emiterArea;
				em.chunksCnt = 0;
				emiters.push_back(em);
				this->positions.reserve(this->positions.capacity() + em.chunkSize * em.maxChunks);
				this->velocities.reserve(this->velocities.capacity() + em.chunkSize * em.maxChunks);
				this->densities.reserve(this->densities.capacity() + em.chunkSize * em.maxChunks);
				this->external_forces.reserve(this->external_forces.capacity() + em.chunkSize * em.maxChunks);
				this->neighbors.reserve(this->neighbors.capacity() + em.chunkSize * em.maxChunks);
				return emiters.size() - 1;
			};
			//!WARNING - incompatiable with killFugutuves
			//TODO - add handling of start pointers relocations in case of kill killFugitives() 
			//	removes particles from the array
			

			void emitParticles( const EmiterId emiterId,
						const Vector3R& extForces,
						const Real wallockTime, 
						NeighborhoodSearch& ns,
						const Vector3R& velocityVector)
			{
				assert(emiterId < emiters.size());
				Emiter& em = emiters[emiterId];
				if(wallockTime - em.prevEmmitionTime < 1.5 * this->diameter / em.emitVelocity.norm()){
					return;
				}
				em.prevEmmitionTime = wallockTime;

				vector<Vector3R> emitedParticlePositions(em.chunkSize);
				Vector3R velocity = velocityVector.normalized() * em.emitVelocity.norm();
				assert(emitedParticlePositions.size() == em.chunkSize);
				sample_emiter_fluid_particles(emitedParticlePositions, 
												velocityVector, 
												this->diameter, 
												em.pos);
				assert(em.chunkSize == emitedParticlePositions.size());
				assert(em.chunksCnt <= em.maxChunks);
				if(em.chunksCnt == em.maxChunks){
					em.chunkOffsets.push_back(em.chunkOffsets.front());
					em.chunkOffsets.pop_front();
				} else if(em.chunksCnt < em.maxChunks){
					em.chunksCnt++;
					em.chunkOffsets.push_back(this->positions.size());
					this->positions.resize(this->positions.size() + em.chunkSize);
					this->velocities.resize(this->velocities.size() + em.chunkSize);
					this->densities.resize(this->densities.size() + em.chunkSize);
					this->external_forces.resize(this->external_forces.size() + em.chunkSize);
					this->neighbors.resize(this->neighbors.size() + em.chunkSize);
					ns.resize_point_set(0, (Real*)this->positions.data(), this->positions.size());
				}
				#pragma omp parallel for schedule(static)
				for (size_t i = 0; i < em.chunkSize; ++i)
				{
					this->positions[em.chunkOffsets.back() + i] = emitedParticlePositions[i];
					this->velocities[em.chunkOffsets.back() + i] = velocity;
					this->densities[em.chunkOffsets.back() + i] = 0;
					this->external_forces[em.chunkOffsets.back() + i] = extForces;
				}
			};

			void emitParticles(const EmiterId emiterId,
						const Vector3R& extForces,
						const Real wallockTime, 
						NeighborhoodSearch& ns){
				emitParticles(emiterId, extForces, wallockTime, ns, emiters[emiterId].emitVelocity);
			}

			void setPositions(vector<Vector3R> &newPositions)
			{
				assert(this->positions.size() == newPositions.size());
				this->positions.assign(newPositions.begin(), newPositions.end());
			}

			void setGravity(Real gravity)
			{
				external_forces.assign(this->size(), this->mass * Vector3R(0.0, gravity, 0.0));
			}

			Real getDiameter() const
			{
				return this->diameter;
			}

			Real getMass() const
			{
				return this->diameter * this->diameter * this->diameter * restDensity;
			}

			Real getSmoothingLength() const
			{
				return this->mEta * this->diameter;
			}

			Real getCompactSupport() const
			{
				return this->mEta * this->diameter * 2;
			}

			const vector<Real>& getDensities() const
			{
				return densities;
			};
			vector<Real>& getDensities()
			{
				return densities;
			};

			const vector<Vector3R>& getVelocities() const
			{
				return velocities;
			};
			
			vector<Vector3R>& getVelocities()
			{
				return velocities;
			};

			const vector<vector<vector<unsigned int> > >& getNeighbors() const
			{
				return neighbors;
			}

			vector<vector<vector<unsigned int> > >& getNeighbors()
			{
				return neighbors;
			}

			const vector<Vector3R>& getExternalForces() const
			{
				return external_forces;
			};

			vector<Vector3R>& getExternalForces()
			{
				return external_forces;
			};

			void findNeighbors(NeighborhoodSearch &ns)
			{
				ns.update_point_sets();

				#pragma omp parallel for schedule(static)
				for(size_t i = 0; i < this->size(); i++) {
					ns.find_neighbors(0, i, neighbors[i]);
				}
			};

			void killFugitives(const Vector3R &lowerCorner,const Vector3R &upperCorner, NeighborhoodSearch &ns)
			{

				#pragma omp parallel for schedule(static)
				for (size_t i = 0; i < this->size(); i ++) {

					bool inside = true;

					inside &= (lowerCorner(0) <= positions[i](0));
					inside &= (lowerCorner(1) <= positions[i](1));
					inside &= (lowerCorner(2) <= positions[i](2));

					inside &= (positions[i](0) <= upperCorner(0));
					inside &= (positions[i](1) <= upperCorner(1));
					inside &= (positions[i](2) <= upperCorner(2));
					if(!inside){
						positions[i] = lowerCorner;
						velocities[i] = Vector3R(0,0,0);
						densities[i] = 0;
						external_forces[i] = Vector3R(0,0,0);
					}
				}
			}

			void clipVelocities(Real capVelo)
			{
				if(capVelo < 0){
					return;
				}
				#pragma omp parallel for schedule(static)
				for (size_t i = 0; i < this->size(); i++){
					if (velocities[i].norm() >= capVelo) {
						velocities[i] = velocities[i].normalized() * capVelo;
					}
				}
			}

			Real getCourantBound()
			{
				Real vMaxNorm = 0.0;

				#pragma omp parallel for reduction(max:vMaxNorm)
				for (size_t i = 0; i < this->size(); i++){
					vMaxNorm = max(velocities[i].norm(), vMaxNorm);
				}

				return 0.5 * this->diameter / vMaxNorm;
			}

			void add_fluid_particles(const vector<Vector3R>& pos, const vector<Vector3R>& vel){
				assert(pos.size() == vel.size());
				this->positions.insert(this->positions.end(), pos.begin(), pos.end());
				this->velocities.insert(this->velocities.end(), vel.begin(), vel.end());
				this->densities.insert(this->densities.end(), pos.size(), 0);
				this->neighbors.resize(this->neighbors.size() + pos.size());
			}
			FluidSystem(vector<Vector3R> &positions, 
						vector<Vector3R> &velocities, 
						vector<Real> &densities, 
						Real restDensity, 
						Real fluidVolume, 
						Real eta):
				ParticleSystem(positions, restDensity),
				mEta(eta)
			{
				this->mass = (this->restDensity * fluidVolume) / this->positions.size();
				this->diameter = cbrt(this->mass / this->restDensity);

				this->densities.assign(densities.begin(), densities.end());
				this->velocities.assign(velocities.begin(), velocities.end());

				this->neighbors.resize(this->size());
			}
			FluidSystem(Real restDensityVal, 
						Real diameterVal,
						Real etaVal):
				ParticleSystem(restDensityVal),
				mass(pow3(diameterVal) * restDensityVal),
				diameter(diameterVal),
				mEta(etaVal)
			{
			}
			FluidSystem(vector<Vector3R>&& positions, 
						vector<Vector3R>&& velocities, 
						vector<Real>&& densities, 
						Real restDensity, 
						Real compactSuport,
						Real eta):
				ParticleSystem(positions, restDensity),
				diameter(compactSuport / eta),
				mEta(eta)
			{
				this->densities.swap(densities);
				this->velocities.swap(velocities);
				this->neighbors.resize(this->size());
			}
			
	};
}
