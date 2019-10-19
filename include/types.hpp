#pragma once
#include <iostream>
#ifdef DEBUG
//#include <boost/stacktrace.hpp>
#endif

using namespace std;

//#define DEBUG
#ifdef DEBUG
#define pr_dbg(str, args...) fprintf(stderr, "[%s,%s,%d\t]: " str "\n", __FILE__, __FUNCTION__, __LINE__, ##args)
#define pr_stack() //std::cerr<< boost::stacktrace::stacktrace()
#else
#define pr_dbg(str, args...)
#define pr_stack()
#endif

#define pr_err(str, args...) fprintf(stderr, "ERROR: " str "\n", ##args)
#define pr_warn(str, args...) fprintf(stderr, "WARNING: " str "\n", ##args)
#define pr_info(str, args...) fprintf(stderr, "INFO: " str "\n", ##args)

#undef assert
#define assert(expr, msg...) do{ \
	if(!(expr)){ \
		pr_dbg("ASSERTION FAILES " #expr); \
		pr_dbg(msg); \
		pr_err("STACK TRACE:"); \
		pr_stack(); \
		abort(); \
	}\
}while(0)

enum opcode {
	STATUS_OK = 0,
	STATUS_NOK
};

using Real = double;

static inline double getRand(const double minVal,const double maxVal){
	assert(maxVal - minVal >= 0.0);
	return minVal + double(rand())/RAND_MAX * (maxVal - minVal);
}
