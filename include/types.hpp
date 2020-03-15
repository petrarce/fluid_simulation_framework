#pragma once
#include <iostream>
#include <Eigen/Dense>

#ifdef PROFILE

#include <stack>
#include <chrono>

static std::stack<std::pair<int, std::chrono::time_point<std::chrono::system_clock>>> profile_stack;

#define PROFILE_START \
do{ \
	profile_stack.push(std::make_pair(__LINE__, std::chrono::system_clock::now())); \
}while(0);

#define PROFILE_STOP \
do{ \
	pair<int, std::chrono::time_point<std::chrono::system_clock>> last_pt = profile_stack.top(); \
	profile_stack.pop(); \
	fprintf(stderr, "[PROFILE (%s:%d-%d)]: %ld ms\n", \
		__FILE__, \
		last_pt.first, \
		__LINE__, \
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - last_pt.second).count()); \
}while(0);

#else
#define PROFILE_START
#define PROFILE_STOP
#endif

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
