#pragma once
#include <types.hpp>
#include <Eigen/Dense>
#include <vector>
#include <CompactNSearch>
#include <kernel.h>
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

				Real compactSupport = diameter * eta;
				NeighborhoodSearch ns(compactSupport, false);

				ns.add_point_set((Real*)(this->positions.data()), this->positions.size(), false);

				ns.update_point_sets();

				vector<vector<vector<unsigned int>> > neighbours;
				neighbours.resize(this->size());

				//foreach point find neighbors
				for(int i = 0; i < this->size(); i++){
					ns.find_neighbors(0, i, neighbours[i]);
				};
				assert(neighbours.size() == 0 || neighbours[0].size() == 1);
				Real avgNghbCnt = 0;
				size_t maxNghbCnt = 0;
				for(int i = 0; i < this->size(); i++){
					avgNghbCnt += neighbours[i][0].size();
					maxNghbCnt = max(maxNghbCnt, neighbours[i][0].size());
				}
				avgNghbCnt /= this->size();
				Real factor = avgNghbCnt / maxNghbCnt;
				set<unsigned int> deleted;
				//for each pont
				for(int i = 0; i < this->size(); i++){
					//if point is in deleted list - skip
					if(deleted.find(i) != deleted.end()){
						continue;
					}
					//find all neighboring points, that are within a threshold
					for(int j : neighbours[i][0]){
						//put all such points into deleted list
						if((this->positions[i] - this->positions[j]).norm() < 0.5 * compactSupport){
							deleted.insert(j);
						}
					}
				}
				//for each point do
				for(int i = 0; i < this->size(); i++){
					//for all neighbors of point do
					if(deleted.find(i) != deleted.end()){
						continue;
					}
								
					Real sum = 0.0;
					for(int j : neighbours[i][0]){
						if(deleted.find(j) != deleted.end()){
							continue;
						}
						sum += kernelFunction(this->positions[i], 
												this->positions[j], 
												0.5 * compactSupport);
					}
					if(sum < threshold){
						this->volumes[i] = pow3(diameter);
						continue;
					}
					this->volumes[i] = 1.0 / sum;

				}
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
			Real smooth_length;
			Real compact_support;

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
				for(int i = 0; i < partCnt; i++){
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
				return emiters.size() - 1;
			};
			//!WARNING - incompatiable with killFugutuves
			//TODO - add handling of start pointers relocations in case of kill killFugitives() 
			//	removes particles from the array
			

			void emit( const EmiterId emiterId,
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
					em.chunkOffsets.push_front(this->positions.size());
					this->positions.insert(this->positions.begin(), 
											emitedParticlePositions.begin(), 
											emitedParticlePositions.end());
					this->velocities.insert(this->velocities.begin(), em.chunkSize, velocity);
					this->densities.insert(this->densities.begin(), em.chunkSize, 0);
					this->external_forces.insert(this->external_forces.begin(), em.chunkSize, this->mass * extForces);
					this->neighbors.resize(this->neighbors.size() + em.chunkSize);
					ns.resize_point_set(0, (Real*)this->positions.data(), this->positions.size());
					return;
				}
				#pragma omp parallel for schedule(static)
				for (int i = 0; i < em.chunkSize; ++i)
				{
					this->positions[em.chunkOffsets.back() + i] = emitedParticlePositions[i];
					this->velocities[em.chunkOffsets.back() + i] = velocity;
					this->external_forces[em.chunkOffsets.back() + i] = extForces;
				}
			};

			void emit(const EmiterId emiterId,
						const Vector3R& extForces,
						const Real wallockTime, 
						NeighborhoodSearch& ns){
				Emiter& em = emiters[emiterId];
				emit(emiterId, extForces, wallockTime, ns, emiters[emiterId].emitVelocity);
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
				return this->mass;
			}

			Real getSmoothingLength() const
			{
				return this->smooth_length;
			}

			Real getCompactSupport() const
			{
				return this->compact_support;
			}

			vector<Real>& getDensities()
			{
				return densities;
			};

			vector<Vector3R>& getVelocities()
			{
				return velocities;
			};

			vector<vector<vector<unsigned int> > >& getNeighbors()
			{
				return neighbors;
			}

			vector<Vector3R>& getExternalForces()
			{
				return external_forces;
			};

			void findNeighbors(NeighborhoodSearch &ns)
			{
				ns.update_point_sets();

				for(int i = 0; i < this->size(); i++) ns.find_neighbors(0, i, neighbors[i]);
			};

			void killFugitives(const Vector3R &lowerCorner,const Vector3R &upperCorner, NeighborhoodSearch &ns)
			{
				vector<size_t> fugitives;

				#pragma omp parallel for schedule(static) shared(fugitives)
				for (size_t i = 0; i < this->size(); i ++) {

					bool inside = true;

					inside &= (lowerCorner(0) <= positions[i](0));
					inside &= (lowerCorner(1) <= positions[i](1));
					inside &= (lowerCorner(2) <= positions[i](2));

					inside &= (positions[i](0) <= upperCorner(0));
					inside &= (positions[i](1) <= upperCorner(1));
					inside &= (positions[i](2) <= upperCorner(2));

					if (!inside) {
						#pragma omp critical
						fugitives.push_back(i);
					}
				}
				if (fugitives.empty()) return;

				std::reverse(fugitives.begin(), fugitives.end());

				for (auto i : fugitives) {
					bool emiter_particle = false;
					for(Emiter& e : emiters){
						for(size_t chunkOffset : e.chunkOffsets){
							if(i >=chunkOffset && i < chunkOffset + e.chunkSize){
								emiter_particle = true;
								positions[i] = e.pos;
								velocities[i] = Vector3R(0,0,0);
								densities[i] = 0;
							}
						}
					}
					if(emiter_particle){
						continue;
					}
					/*no need to reload offsets for emmiter particles a.s.a. all emiter 
						particles are resided at the end of the data arrays*/
					positions[i] = positions.back();
					densities[i] = densities.back();
					velocities[i] = velocities.back();
					external_forces[i] = external_forces.back();

					positions.pop_back();
					densities.pop_back();
					velocities.pop_back();
					external_forces.pop_back();
				}
				ns.resize_point_set(0, (Real*)positions.data(), positions.size());

				neighbors.resize(this->size());
			}

			void clipVelocities(Real capVelo)
			{
				if(capVelo < 0){
					return;
				}
				#pragma omp parallel for schedule(static)
				for (int i = 0; i < this->size(); i++){
					if (velocities[i].norm() >= capVelo) {
						velocities[i] = velocities[i].normalized() * capVelo;
					}
				}
			}

			Real getCourantBound()
			{
				Real vMaxNorm = 0.0;

				#pragma omp parallel for reduction(max:vMaxNorm)
				for (int i = 0; i < this->size(); i++){
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
			FluidSystem(vector<Vector3R> &positions, vector<Vector3R> &velocities, vector<Real> &densities, Real restDensity, Real fluidVolume, Real eta):ParticleSystem(positions, restDensity)
			{
				this->mass = (this->restDensity * fluidVolume) / this->positions.size();
				this->diameter = cbrt(this->mass / this->restDensity);
				this->smooth_length = eta * this->diameter;
				this->compact_support = 2.0 * this->smooth_length;

				this->densities.assign(densities.begin(), densities.end());
				this->velocities.assign(velocities.begin(), velocities.end());

				this->neighbors.resize(this->size());
			};
			FluidSystem(Real restDensityVal, Real diameterVal, Real etaVal):ParticleSystem(restDensityVal),
				diameter(diameterVal),
				mass(pow3(diameterVal) * restDensityVal),
				smooth_length(diameterVal * etaVal),
				compact_support(2*diameterVal * etaVal)
			{
			};
	};
};
