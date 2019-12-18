#pragma once
#include <types.hpp>
#include <Eigen/Dense>
#include <vector>
#include <CompactNSearch>
#include <kernel.h>
#include <set>

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
				fprintf(stderr, "factor = %f", factor);
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
			Real mass;
			Real diameter;
			Real smooth_length;
			Real compact_support;

			vector<Real> densities;
			vector<Vector3R> velocities;
			vector<Vector3R> external_forces;

			vector<vector<vector<unsigned int> > > neighbors;

		public:
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

			void killFugitives(Vector3R &lowerCorner, Vector3R &upperCorner, NeighborhoodSearch &ns)
			{
				vector<size_t> fugitives;

				for (size_t i = 0; i < this->size(); i ++) {

					bool inside = true;

					inside &= (lowerCorner(0) <= positions[i](0));
					inside &= (lowerCorner(1) <= positions[i](1));
					inside &= (lowerCorner(2) <= positions[i](2));

					inside &= (positions[i](0) <= upperCorner(0));
					inside &= (positions[i](1) <= upperCorner(1));
					inside &= (positions[i](2) <= upperCorner(2));

					if (!inside) fugitives.push_back(i);
				}
				if (fugitives.empty()) return;

				std::reverse(fugitives.begin(), fugitives.end());

				for (auto i : fugitives) {

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
				for (int i = 0; i < this->size(); i++) if (velocities[i].norm() >= capVelo) velocities[i] = velocities[i].normalized() * capVelo;
			}

			Real getCourantBound()
			{
				Real vMaxNorm = 0.0;

				for (int i = 0; i < this->size(); i++) vMaxNorm = max(velocities[i].norm(), vMaxNorm);

				return 0.5 * this->diameter / vMaxNorm;
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
	};
};
