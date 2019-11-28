#pragma once
#include <iostream>
#include <Eigen/Dense>

using namespace std;

#ifdef DEBUG
#define pr_dbg(str, args...) fprintf(stderr, "[%s,%s,%d\t]: " str "\n", __FILE__, __FUNCTION__, __LINE__, ##args)
#define pr_stack()
#else
#define pr_dbg(str, args...)
#define pr_stack()
#endif

#define pr_err(str, args...) fprintf(stderr, "ERROR: " str "\n", ##args)
#define pr_warn(str, args...) fprintf(stderr, "WARNING: " str "\n", ##args)
#define pr_info(str, args...) fprintf(stderr, "INFO: " str "\n", ##args)

using Real = double;
using Vector3R = Eigen::Vector3d;

constexpr Real threshold = 1e-6;

static inline double getRand(const double minVal,const double maxVal){
	assert(maxVal - minVal >= 0.0);
	return minVal + double(rand())/RAND_MAX * (maxVal - minVal);
}
