#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Dense>

using namespace std;

Eigen::MatrixXi computeKernel(const Eigen::MatrixXi& kernel, int iterations, int initialKernelSize)
{
	assert(kernel.rows() >= 3);
	assert((kernel.rows()%2) == 1);
	assert(kernel.rows() == kernel.cols());
	Eigen::MatrixXi oldKernel = kernel;

	for(int k = 0; k < iterations; k++ )
	{
		Eigen::MatrixXi newKernel(oldKernel.rows() + 2, oldKernel.rows() + 2);
		for(int i = -newKernel.rows()/2; i <= newKernel.rows()/2; i++)
		{
			for(int j = -newKernel.rows()/2; j <= newKernel.rows()/2; j++)
			{
				int iInd = newKernel.rows()/2 + i;
				int jInd = newKernel.rows()/2 + j;
				for(int vo = -initialKernelSize; vo <= initialKernelSize; vo ++)
				{
					for(int ho = -initialKernelSize; ho <= initialKernelSize; ho++)
					{
						int iOffsInd = oldKernel.rows()/2 + i + vo;
						int jOffsInd = oldKernel.rows()/2 + j + ho;
						if(iOffsInd < 0 || jOffsInd < 0 || iOffsInd >= oldKernel.rows() || jOffsInd >= oldKernel.rows())
							continue;

						newKernel(iInd, jInd) += oldKernel(iOffsInd, jOffsInd);
					}
				}
			}
		}
		oldKernel = newKernel;
	}
	return oldKernel;
}


int main(int ac, char** av)
{
	Eigen::MatrixXi kernel(3,3); //= Eigen::MatrixXi::Once(3,3);
	kernel << 1,1,1,
			1,1,1,
			1,1,1;
	auto computedKernel = computeKernel(kernel, stoi(av[1]), 1);
	std::cout << computedKernel << std::endl;
	std::cout << "matsum: " <<  computedKernel.sum() / pow(9, stoi(av[1])+ 1);

}