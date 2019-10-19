#include <types.hpp>
#include <vector>

using namespace std;

class NBS{
private:
	Real cs;

public:
	NBS(Real r);
	~NBS();
	
	/*
	 * returns list of neighbours for each point in the set
	 * data - pointer to data set
	 * data_size - number of entries in data set
	 * neighbor_list returned list of indexes of neighbour particles in data set for each particle
	*/
	opcode set_compack_support(Real r);
	opcode bf_find_neighbours(Real const* data, const size_t data_size, 
		vector<vector<unsigned int>> &neighbor_list);
};