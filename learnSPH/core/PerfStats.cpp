#include "PerfStats.hpp"

boost::timer::cpu_times operator+(const boost::timer::cpu_times& left, const boost::timer::cpu_times& right)
{
	boost::timer::cpu_times total = left;
	total.wall += right.wall;
	total.user += right.user;
	total.system += right.system;
	return total;
}

void PerfStats::startTimer(const std::string& stats)
{
	auto timer = mTimers.find(stats);
	if(timer == mTimers.end())
	{
		mTimers[stats] = boost::timer::cpu_timer();
		timer = mTimers.find(stats);
		timer->second.start();
	}
	else
	{
		if(!timer->second.is_stopped())
			pr_warn("timer for %s was not stopped before restarting", timer->first.c_str());
		timer->second.resume();
	}
}
void PerfStats::stopTimer(const std::string& stats)
{
	auto timer = mTimers.find(stats);

	if(timer == mTimers.end())
		std::runtime_error("timer for " + stats + "was not started before stopping");

	timer->second.stop();
	boost::timer::cpu_times elapsed = timer->second.elapsed();
	auto statsData = mStats.find(stats);
	if(statsData == mStats.end())
	{
		mStats[stats] = boost::timer::cpu_times();
		statsData = mStats.find(stats);
	}
	assert(statsData != mStats.end());
	statsData->second = statsData->second + elapsed;


}
void PerfStats::PrintStatistics(const std::string& metadata)
{
	std::cout << "=====PERFORMANCE STATISTICS=====\n";
	std::cout << "METADATA: " << metadata << "\n";
	for(const auto& sd : mTimers)
	{
		if(!sd.second.is_stopped())
			pr_warn("Timer for %s was either not started or was not stopped", sd.first.c_str());

		std::cout << sd.first + "\t&\t" << sd.second.format(6, "%w") << std::endl;
	}
	std::cout << "=====END PERFORMANCE STATISTICS=====\n";
}
