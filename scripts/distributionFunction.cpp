#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <unordered_set>

using namespace std;

int main(int argc, char** argv)
{
	mt19937 gen(1);
	vector<size_t> initialDistribution(stoi(argv[1]));
	vector<size_t> resultingDistribution(stoi(argv[1]));
	size_t lowerBound = 0;
	size_t upperBound = resultingDistribution.size()*resultingDistribution.size() - 1;
	unordered_set<size_t> acceptedIndices;

	for(int i = 0; i < stoi(argv[2]); i++)
	{
		float value = gen() / static_cast<float>(gen.max());
		assert(value >= 0 && value <= 1);
		size_t index = initialDistribution.size() * initialDistribution.size();
		size_t level = sqrt(index * value);
		assert(level < initialDistribution.size());
		initialDistribution[level]++;



		index = lowerBound + value * value * (upperBound - lowerBound);				
		size_t currentIndex = index;
		while(acceptedIndices.count(currentIndex)){
			if(currentIndex == lowerBound)
				currentIndex = upperBound;
			else
				currentIndex--;
			assert(currentIndex != index);
		}
		acceptedIndices.insert(currentIndex);
		level = sqrt(currentIndex);
		assert(level < initialDistribution.size());
		resultingDistribution[level]++;
		if(currentIndex == lowerBound)
		{
			currentIndex++;
			while(acceptedIndices.count(currentIndex))
			{
				currentIndex++;
				assert(currentIndex < upperBound);
			}
			lowerBound = currentIndex;
		}
		else if(currentIndex == upperBound)
		{
			currentIndex--;
			while(acceptedIndices.count(currentIndex))
			{
				currentIndex--;
				assert(currentIndex > lowerBound);
			}
			upperBound = currentIndex;
		}
		assert(upperBound > lowerBound);

	}

	cout << "initial distribution\n";
	for(int i = 0; i < initialDistribution.size(); i++)
		cout << "id: level " << i << " probability: " << initialDistribution[i] / stof(argv[2]) << endl;

	cout << "final distribution\n";
	for(int i = 0; i < resultingDistribution.size(); i++)
		cout << "rd: level " << i << " probability: " << resultingDistribution[i] / stof(argv[2]) << endl;

	return 0;

}