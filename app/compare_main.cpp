#include <iostream>
#include <CompactNSearch.h>
#include <kernel.h>
#include <NBsearch.hpp>
#include <types.hpp>
#include <chrono>

using namespace std;
using namespace CompactNSearch;
using namespace learnSPH::kernel;



int main(int argc, char** argv){

	int number_of_points = 0;
	Real compact_support = 0;
	int iterations = 0;
	try{
		number_of_points = stoi(argv[1]);
		compact_support = stod(argv[2]);
		iterations = stoi(argv[3]);
	}
	catch(const invalid_argument& e){
		pr_dbg("invalid argument: %s", e.what());
	}
 	vector<array<Real, 3>> points;
 	vector<vector<unsigned int>> neighbours;

 	for(int i = 0, npoints = number_of_points; 
 			i < iterations; 
 			i++, npoints += number_of_points)
 	{
		points.clear();
		points.reserve(npoints);
		for(int i = 0; i < npoints; i ++){
			Real x = getRand(-1,1);
			Real y = getRand(-1,1);
			Real z = getRand(-1,1);
			points.push_back({x,y,z});
		}

		//perform brut force Neighbourhood search
		NBS custom_nbs(compact_support);
		auto t0 = std::chrono::high_resolution_clock::now();
		custom_nbs.bf_find_neighbours(points.front().data(), points.size(), neighbours);
		auto t1 = std::chrono::high_resolution_clock::now();
		
		fprintf(stderr, "custom_nbs takes %ld ms on %d particles\n", 
			std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(), npoints);

		//perform neighbourhood search using CompactNsearch library
		NeighborhoodSearch nsearch(compact_support, true);
		nsearch.add_point_set(points.front().data(), points.size(), true, true);
		nsearch.z_sort();
		for (auto i = 0u; i < nsearch.n_point_sets(); ++i)
		{
			auto const& d = nsearch.point_set(i);
			d.sort_field(points.data());

		}
		t0 = std::chrono::high_resolution_clock::now();
		nsearch.find_neighbors();
		fprintf(stderr, "CompactNSearch takes %ld ms on %d particles\n", 
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - t0).count(), npoints);
	}

	return 0;
}