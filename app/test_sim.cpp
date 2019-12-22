#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>

#include <Eigen/Dense>
#include <storage.h>
#include <types.hpp>
#include <particle_sampler.h>
#include <CompactNSearch>

#include <vtk_writer.h>
#include <solver.h>
#include <chrono>

#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

#include <boost/thread/thread.hpp>
#include <blocking_queue.h>

using namespace CompactNSearch;
using namespace learnSPH;
using namespace boost::program_options;


void save_vectors(const std::string &path, std::vector<Vector3R> &data)
{
	std::ofstream os(path, std::ios::binary);

	cereal::BinaryOutputArchive boa(os);

	boa(data.size());

	for (Vector3R &vec : data) boa(vec(0), vec(1), vec(2));
}

void save_scalars(const std::string &path, std::vector<Real> &data)
{
	std::ofstream os(path, std::ios::binary);

	cereal::BinaryOutputArchive boa(os);

	boa(data.size());

	for (auto val : data) boa(val);
}

struct {
	Real render_ts;
	Real lower_bound_ts;
	Real viscosity;
	Real fluid_rest_density;
	Real border_rest_density;
	Real friction;
	size_t pbfIterations;
	string border_model_path;
	string sim_name;
	string outp_dir_path;
	Real sim_duration;
	Real preasureCoefficient;
	Real sampling_distance;
	Real eta;
	Real max_velocity;
	bool dbg;
	Vector3R clip_lower_bound;
	Vector3R clip_upper_bound;
} cmdValues;



typedef struct {
	Vector3R lower_corner;
	Vector3R upper_corner;
} FluidDisplacement;

typedef struct {
	Vector3R em_pos;
	Vector3R em_veloc;
	Vector3R em_area;
} EmiterDisplacement;

static void save_current_state(const string& dir_path, const string& sim_name, size_t frame, FluidSystem& fluid){
		string filename = dir_path + sim_name + '_' + std::to_string(frame) + ".vtk";
		learnSPH::saveParticlesToVTK(filename, fluid.getPositions(), fluid.getDensities(), fluid.getVelocities());
		vector<Real> params;
		params.push_back(fluid.getCompactSupport());
		params.push_back(fluid.getSmoothingLength());
		params.push_back(fluid.getMass());
		params.push_back(fluid.getRestDensity());
		filename = dir_path + sim_name + "_params_" + std::to_string(frame) + ".cereal";
		save_scalars(filename, params);
		filename = dir_path + sim_name + "_positions_" + std::to_string(frame) + ".cereal";
		save_vectors(filename, fluid.getPositions());
		filename = dir_path + sim_name + "_densities_" + std::to_string(frame) + ".cereal";
		save_scalars(filename, fluid.getDensities());
}

static void parse_fluid_displacement(const string& str, FluidDisplacement& fd)
{
  	typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
  	boost::char_separator<char> sep{" (),[]"};
	tokenizer tok{str, sep};
	char counter = 0;
	invalid_argument exeption("incorrect format: " + str);
	try{
		for(string v : tok){
			if(counter < 3){
				fd.lower_corner[counter] = stod(v);
			} else if(counter < 6){
				fd.upper_corner[counter-3] = stod(v);
			} else{
				throw invalid_argument(""); 
			}
			counter++;
		}
		if(counter < 6){
			throw invalid_argument("");
		}
	} catch(const invalid_argument& e){
		throw exeption;
	}
}

static void validate_cmd_options(const variables_map& vm)
{
	if(!(vm.count("render-ts") &&
			vm.count("border-model-path") &&
			vm.count("fluid-displacement") &&
			vm.count("sim-duration") &&
			vm.count("sim-type") &&
			vm.count("sample-dist") &&
			vm.count("clip-area"))){
		cout << "next options are mandatory: --render-ts --border-model-path --sim-duration --sim-type --sample-dist --fluid-displacement --clip-area" << endl;
		throw invalid_argument("some options are missed");
	}
}

static void assign_cmd_options(const variables_map& vm){
	cmdValues.render_ts = vm["render-ts"].as<Real>();
	cmdValues.lower_bound_ts = (vm.count("lower-bound-ts"))?vm["lower-bound-ts"].as<Real>():cmdValues.render_ts;
	cmdValues.viscosity = vm["viscosity"].as<Real>();
	cmdValues.fluid_rest_density = vm["rest-density"].as<Real>();
	cmdValues.border_rest_density = vm["border-rest-density"].as<Real>();
	cmdValues.friction = vm["friction"].as<Real>();
	cmdValues.pbfIterations = vm["iterations"].as<size_t>();
	cmdValues.border_model_path = vm["border-model-path"].as<string>();
	cmdValues.sim_duration = vm["sim-duration"].as<Real>();
	cmdValues.preasureCoefficient = vm["pressure-coeff"].as<Real>();
	cmdValues.sampling_distance = vm["sample-dist"].as<Real>();
	cmdValues.sim_name = vm["sim-name"].as<string>();
	cmdValues.eta = vm["eta"].as<Real>();
	cmdValues.outp_dir_path = vm["output-directory"].as<string>();
	cmdValues.max_velocity = vm["max-velocity"].as<Real>();
	cmdValues.dbg = (vm.count("dbg"))?true:false;
	FluidDisplacement fd;
	parse_fluid_displacement(vm["clip-area"].as<string>(), fd);
	cmdValues.clip_lower_bound = fd.lower_corner;
	cmdValues.clip_upper_bound = fd.upper_corner;
}

static int generate_simulation_frame_PBF(FluidSystem& fluid, NeighborhoodSearch& ns, BorderSystem& border)
{
	Real cur_sim_time = 0.0;
	int physical_steps = 0;
	fluid.findNeighbors(ns);
	while (cur_sim_time < cmdValues.render_ts) {
		learnSPH::calculate_dencities((&fluid), (&border));
		
		vector<Vector3R> accelerations(fluid.size(), Vector3R(0.0, 0.0, 0.0));
		learnSPH::add_visco_component(accelerations, (&fluid), (&border), cmdValues.viscosity, cmdValues.friction);
		learnSPH::add_exter_component(accelerations, (&fluid));
		
		Real update_step = max(cmdValues.lower_bound_ts, min(fluid.getCourantBound(), cmdValues.render_ts));
		auto positions = fluid.getPositions();
		learnSPH::smooth_symplectic_euler(accelerations, (&fluid), 0.5, update_step);
		
		fluid.findNeighbors(ns);
		learnSPH::correct_position((&fluid), (&border), positions, update_step, cmdValues.pbfIterations);
		
		fluid.killFugitives(cmdValues.clip_lower_bound, cmdValues.clip_upper_bound, ns);
		fluid.clipVelocities(cmdValues.max_velocity);
		
		cur_sim_time += update_step;
		physical_steps ++;
		if(cmdValues.dbg){
			break;
		}

	}
	return physical_steps;
}

//queue in which simulation states resided
BlockingQueue<FluidSystem*> simulationStateQueue(5);
boost::mutex print_lock;

void gen_frames_thread(FluidSystem& fluid, NeighborhoodSearch& ns,BorderSystem& border, int n_frames)
{
	for (int frame = 1; frame <= n_frames; frame ++) {
		int physical_steps = generate_simulation_frame_PBF(fluid, ns, border);
		fprintf(stderr, "[%d] physical updates were carried out for rendering frame [%d]/[%d]\n",
				physical_steps, frame, n_frames);
		FluidSystem* curFluidState = new FluidSystem(fluid);
		assert(curFluidState != 0);

		simulationStateQueue.push(curFluidState);
#ifdef DEBUG
		boost::mutex::scoped_lock(print_lock);
		fprintf(stderr, "pushed new simulation state\n");
#endif
	}
	while(!simulationStateQueue.empty()){};
	simulationStateQueue.close();

}

void save_sim_state_thread(){
	size_t frame_num = 1;
	while(!simulationStateQueue.closed()){
		FluidSystem* fluid;
		if(!simulationStateQueue.pop(fluid)){
			break;
		}
		save_current_state(cmdValues.outp_dir_path, cmdValues.sim_name, frame_num, *fluid);
#ifdef DEBUG
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
		boost::mutex::scoped_lock(print_lock);
		fprintf(stderr, "saved simulation state\n");
#endif
		frame_num++;
		delete fluid;
	}
}
int main(int ac, char** av)
{
	options_description po_def("Fluid simulator cmd arguments description");
	po_def.add_options()
		("help,h", "print this message\n")
		("render-ts,r", value<Real>(), "simulation time-step, at which simulation state will be saved to vtk/cereal\n")
		("lower-bound-ts,l", value<Real>(), "lower bound at which simulation time can decreased\n")
		("emitter-displacement,e", value<vector<string>>(), "string, which specifies where emitter should be placed inside the simulation\n"
										"\tto specify emitter position, emit area and emit velocity use next format: "
										"\"[\%f,\%f,\%f],\%f,[\%f,\%f,\%f]\"\n")
		("viscosity,v", value<Real>()->default_value(1e-2), "fluid viscosity value\n")
		("rest-density,d", value<Real>()->default_value(1e+3), "fluid rest density\n")
		("border-rest-density,b", value<Real>()->default_value(3e+3), "border rest density\n")
		("friction,f", value<Real>()->default_value(1e-2), "fluid friction parameter\n")
		("iterations,r", value<size_t>()->default_value(3), "number iterations for position correction\n")
		("border-model-path,i", value<string>(), "path to obj file, where border model is resided\n")
		("sim-duration,s", value<Real>(), "simulation duration\n")
		("fluid-displacement,u", value<vector<string>>(), "specify lower and upper corner of fluid cube, that should be sampled for simulation\n"
											"\tplease use next format to define fluid cube: "
											"\"[\%f,\%f,\%f],[\%f,\%f,\%f]\"\n")
		("sim-type,t", value<string>(), "simulation method type. Next variants are possible: PBF, WCF\n")
		("sample-dist,a", value<Real>(), "particle sampling distance\n")
		("pressure-coeff,p", value<Real>()->default_value(400), "pressure coefficient for WCSPH\n")
		("eta", value<Real>()->default_value(1.2), "eta value - multiplier for compact support and smoothing length\n")
		("sim-name,n", value<string>()->default_value("new_simulation"), "simulation name."
														"\n\tAll vtk and cereal files will use simulation name prefix")
		("output-directory,o", value<string>()->default_value("./"), "specify output directory, where all cereal and vtk files will be saved\n")
		("clip-area,k", value<string>(), "specify area, outside of which fluid particles will be removed from the simulation")
		("max-velocity", value<Real>()->default_value(50.0f), "specify maximum velocity, which will be given to each particle (negative value means no velocity boundary)")
		("dbg", "debug option. If enabled then each physical update will be saved into vtk file");

	//parse and assign command line options
	variables_map vm;
	store(parse_command_line(ac, av, po_def), vm);
	if(vm.count("help")){
		cout << po_def << endl;
		return 0;
	}
	validate_cmd_options(vm);
	assign_cmd_options(vm);

	//generate fluid particles
	FluidSystem fluid(cmdValues.fluid_rest_density, cmdValues.sampling_distance, cmdValues.eta);
	for(const string& fluid_disp : vm["fluid-displacement"].as<vector<string>>()){
		FluidDisplacement fd;
		parse_fluid_displacement(fluid_disp, fd);
		vector<Vector3R> positions;
		sample_fluid_cube(positions, fd.lower_corner, fd.upper_corner, cmdValues.sampling_distance);
		vector<Vector3R> velocity(positions.size(), Vector3R(0,0,0));
		fluid.add_fluid_particles(positions, velocity);
	}
	fluid.setGravity(-9.7);

	//generate border particles from border obj file
	vector<Vector3R> positions;
	sample_border_model_surface(positions, Matrix4d::Identity(), cmdValues.border_model_path, cmdValues.sampling_distance);
	BorderSystem border(positions, cmdValues.border_rest_density, cmdValues.sampling_distance, cmdValues.eta);

	cout << "Number of fluid particles: " << fluid.size() << endl;
	cout << "Number of border particles: " << border.size() << endl;
	
	//create neighborhood search container
	NeighborhoodSearch ns(fluid.getCompactSupport());
	ns.add_point_set((Real*)fluid.getPositions().data(), fluid.size(), true);
	ns.add_point_set((Real*)border.getPositions().data(), border.size(), false);
	
	int n_frames = cmdValues.sim_duration / cmdValues.render_ts;
	cout << "the simulation lasts [" << cmdValues.sim_duration << "] seconds "
			<< "consisting of [" << n_frames << "] frames. "
			<< "a frame is rendered every [" << cmdValues.render_ts << "] seconds" << endl;

	save_current_state(cmdValues.outp_dir_path, cmdValues.sim_name, 0, fluid);
	
	string filename = cmdValues.outp_dir_path + cmdValues.sim_name + "_border.vtk";
	saveParticlesToVTK(filename, border.getPositions(), border.getVolumes(), vector<Vector3R>(border.size()));
	//start simulation
	cout << endl;
	boost::thread simGen(gen_frames_thread, boost::ref(fluid), boost::ref(ns), boost::ref(border), n_frames);
	boost::thread simSaveState(save_sim_state_thread);

	simGen.join();
	simSaveState.join();
	cout << "simulation finished"<<endl;

	cout << endl;
	return 0;
}
