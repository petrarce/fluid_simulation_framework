#include <iostream>
#include <map>
#include <boost/timer/timer.hpp>
#include <types.hpp>

//WARNING: statistics collection is not thread safe!!!
class PerfStats
{
private:
	std::map<std::string, boost::timer::cpu_timer> mTimers;
	std::map<std::string, boost::timer::cpu_times> mStats;
public:
	void startTimer(const std::string& stats);
	void stopTimer(const std::string& stats);
	void PrintStatistics(const std::string& metadata = "");
};

extern PerfStats globalPerfStats;
