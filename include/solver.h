#pragma once
#include <data_set.h>
#include <vector>

using namespace std;
using namespace learnSPH;

namespace learnSPH{
	class Solver {
	private:
	public:
		static opcode calculate_dencities(NormalPartDataSet& normalParticles,
			const BorderPartDataSet& borderParticles,
			const vector<vector<vector<unsigned int>>>& normalParticleNeighbours);
	
	private:
		//deleted constructors and operator=
		Solver();
		Solver(const Solver&);
		~Solver();
		const Solver& operator=(const Solver&){return *this;};
	};
};