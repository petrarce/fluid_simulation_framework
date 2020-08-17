#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>

#include <Eigen/Dense>
#include <storage.h>
#include <types.hpp>
#include <learnSPH/core/particle_sampler.h>
#include <CompactNSearch>

#include <learnSPH/core/vtk_writer.h>
#include <learnSPH/simulation/solver.h>
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
	Real prStiffness;
	Real sampling_distance;
	Real border_sampling_distance;
	Real eta;
	Real max_velocity;
	bool dbg;
	Vector3R clip_lower_bound;
	Vector3R clip_upper_bound;
	string sim_type;
	bool samplingSheme;
	bool smoothSimplectirEuler;
	Real pbfVelocityMultiplier;
	Real cohesion;
	Real adhesion;
	Real gravity;

	void print()
	{
		fprintf(stdout, "Simulation parameters:\n");
		fprintf(stdout, "\trender_ts=%f\n", render_ts);
		fprintf(stdout, "\tlower_bound_ts=%f\n", lower_bound_ts);
		fprintf(stdout, "\tviscosity=%f\n", viscosity);
		fprintf(stdout, "\tfluid_rest_density=%f\n", fluid_rest_density);
		fprintf(stdout, "\tborder_rest_density=%f\n", border_rest_density);
		fprintf(stdout, "\tfriction=%f\n", friction);
		fprintf(stdout, "\tpbfIterations=%lu\n", pbfIterations);
		fprintf(stdout, "\tborder_model_path=%s\n", border_model_path.c_str());
		fprintf(stdout, "\tsim_name=%s\n", sim_name.c_str());
		fprintf(stdout, "\toutp_dir_path=%s\n", outp_dir_path.c_str());
		fprintf(stdout, "\tsim_duration=%f\n", sim_duration);
		fprintf(stdout, "\tprStiffness=%f\n", prStiffness);
		fprintf(stdout, "\tsampling_distance=%f\n", sampling_distance);
		fprintf(stdout, "\border_sampling_distance=%f\n", border_sampling_distance);
		fprintf(stdout, "\teta=%f\n", eta);
		fprintf(stdout, "\tmax_velocity=%f\n", max_velocity);
		fprintf(stdout, "\tdbg=%s\n", dbg?"true":"false");
		fprintf(stdout, "\tclip_lower_bound=[%f,%f,%f]\n", clip_lower_bound[0], clip_lower_bound[1], clip_lower_bound[2]);
		fprintf(stdout, "\tclip_upper_bound=[%f,%f,%f]\n", clip_upper_bound[0], clip_upper_bound[1], clip_upper_bound[2]);
		fprintf(stdout, "\tsim_type=%s\n", sim_type.c_str());
		fprintf(stdout, "\tsamplingSheme=%s\n", samplingSheme?"hexagonal":"nonhexaonal");
		fprintf(stdout, "\tintegration scheme=%s\n", smoothSimplectirEuler?"smooth simplectic euler":"simplectic euler");
		fprintf(stdout, "\tpdf velocity multiplier=%f\n", pbfVelocityMultiplier);
		fprintf(stdout, "\tcohesion=%f\n", cohesion);
		fprintf(stdout, "\tadhesion=%f\n", adhesion);
		fprintf(stdout, "\tgravity=%f\n", gravity);
	}
} cmdValues;



typedef struct {
	Vector3R lower_corner;
	Vector3R upper_corner;
} FluidDisplacement;

typedef struct {
	Vector3R em_pos;
	Vector3R em_veloc;
	Real em_area;
	int max_particles;
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

static void parse_emiter_displacement(const string& str, EmiterDisplacement& ed)
{
  	typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
  	boost::char_separator<char> sep{" (),[]"};
	tokenizer tok{str, sep};
	char counter = 0;
	invalid_argument exeption("incorrect format: " + str);
	try{
		for(string v : tok){
			if(counter < 3){
				ed.em_pos[counter] = stod(v);
			} else if(counter == 3){
				ed.em_area = stod(v);
			}else if(counter < 7){
				ed.em_veloc[counter-4] = stod(v);
			} else if(counter == 7){
				ed.max_particles = stoi(v);
			} else {
				throw invalid_argument(""); 
			}
			counter++;
		}
		if(counter < 8){
			throw invalid_argument("");
		}
	} catch(const invalid_argument& e){
		throw exeption;
	}
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
	vector<string> requiredOptions = {
		"render-ts",
		"border-model-path",
		"sim-duration",
		"sim-type",
		"sample-dist",
		"clip-area"
	};
	for(string opt : requiredOptions){
		if(!vm.count(opt)){
			throw invalid_argument("--" + opt + " option is missing."+
									"\nPlease specify value for this option explicitly");
		}
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
	cmdValues.prStiffness = vm["pressure-stiffness"].as<Real>();
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
	cmdValues.sim_type = vm["sim-type"].as<string>();
	cmdValues.samplingSheme = (vm["sampling-type"].as<string>() == "hex")?true:false;
	cmdValues.smoothSimplectirEuler = (vm["integration-scheme"].as<string>() == "smooth")?true:false;
	cmdValues.border_sampling_distance = (vm.count("border-sample-dist"))?vm["border-sample-dist"].as<Real>():cmdValues.sampling_distance;
	cmdValues.pbfVelocityMultiplier = vm["pbf-velocity-multiplier"].as<Real>();
	cmdValues.cohesion = vm["cohesion"].as<Real>();
	cmdValues.adhesion = vm["adhesion"].as<Real>();
	cmdValues.gravity = vm["gravity"].as<Real>();
}

float wallclock_time = 0;

static int generate_simulation_frame_PBF(FluidSystem& fluid, NeighborhoodSearch& ns, BorderSystem& border)
{
	Real cur_sim_time = 0.0;
	int physical_steps = 0;
	for(int i = 0; i < fluid.emiters_size(); i++){
		fluid.emit(i, Vector3R(0, cmdValues.gravity, 0), wallclock_time, ns);
	}		
	vector<Vector3R> accelerations(fluid.size(), Vector3R(0.0, 0.0, 0.0));
	fluid.findNeighbors(ns);
	while (cur_sim_time < cmdValues.render_ts) {

		learnSPH::calculate_dencities((&fluid), (&border));
		
		learnSPH::add_visco_component(accelerations, (&fluid), (&border), cmdValues.viscosity, cmdValues.friction);
		learnSPH::add_exter_component(accelerations, (&fluid));
		learnSPH::add_surface_tension_component(accelerations, 
												(&fluid), 
												(&border), 
												cmdValues.cohesion, 
												cmdValues.adhesion);
		Real update_step = max(cmdValues.lower_bound_ts, min(fluid.getCourantBound(), cmdValues.render_ts));
		auto positions = fluid.getPositions();
		if(cmdValues.smoothSimplectirEuler){
			learnSPH::smooth_symplectic_euler(accelerations, (&fluid), 0.5, update_step);
		} else {
			learnSPH::symplectic_euler(accelerations, (&fluid), update_step);
		}
		
		fluid.findNeighbors(ns);
		learnSPH::correct_position(
			(&fluid), 
			(&border), 
			positions, 
			update_step, 
			cmdValues.pbfIterations, 
			cmdValues.pbfVelocityMultiplier);
		
		fluid.killFugitives(cmdValues.clip_lower_bound, cmdValues.clip_upper_bound, ns);
		fluid.clipVelocities(cmdValues.max_velocity);
		
		cur_sim_time += update_step;
		physical_steps ++;
		wallclock_time += update_step;
		if(cmdValues.dbg){
			break;
		}

	}
	return physical_steps;
}

static int generate_simulation_frame_EXT(FluidSystem& fluid, NeighborhoodSearch& ns, BorderSystem& border)
{
	Real cur_sim_time = 0.0;
	int physical_steps = 0;
	for(int i = 0; i < fluid.emiters_size(); i++){
		fluid.emit(i, Vector3R(0, cmdValues.gravity, 0), wallclock_time, ns);
	}		
	while (cur_sim_time < cmdValues.render_ts) {
		fluid.findNeighbors(ns);
		learnSPH::calculate_dencities((&fluid), (&border));
		
		vector<Vector3R> accelerations(fluid.size(), Vector3R(0.0, 0.0, 0.0));
		learnSPH::add_visco_component(accelerations, (&fluid), (&border), cmdValues.viscosity, cmdValues.friction);
		learnSPH::add_exter_component(accelerations, (&fluid));
		learnSPH::add_press_component(accelerations, (&fluid), (&border), cmdValues.prStiffness);
		learnSPH::add_surface_tension_component(accelerations, 
												(&fluid), 
												(&border), 
												cmdValues.cohesion, 
												cmdValues.adhesion);
		
		Real update_step = max(cmdValues.lower_bound_ts, min(fluid.getCourantBound(), cmdValues.render_ts));
		auto positions = fluid.getPositions();
		if(cmdValues.smoothSimplectirEuler){
			learnSPH::smooth_symplectic_euler(accelerations, (&fluid), 0.5, update_step);
		} else {
			learnSPH::symplectic_euler(accelerations, (&fluid), update_step);
		}
		
		fluid.killFugitives(cmdValues.clip_lower_bound, cmdValues.clip_upper_bound, ns);
		fluid.clipVelocities(cmdValues.max_velocity);
		
		cur_sim_time += update_step;
		physical_steps ++;
		wallclock_time += update_step;
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
		int physical_steps;
		if(cmdValues.sim_type == "PBF"){
			physical_steps = generate_simulation_frame_PBF(fluid, ns, border);
		} else if(cmdValues.sim_type == "EXT"){
			physical_steps = generate_simulation_frame_EXT(fluid, ns, border);
		} else {
			throw invalid_argument("unknown simulation type: " + cmdValues.sim_type + ". Use --help to check supported");
		}
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
		("help", "print this message\n")
		("render-ts", value<Real>(), "simulation time-step, at which simulation state will be saved to vtk/cereal\n")
		("lower-bound-ts", value<Real>(), "lower bound at which simulation time can decreased\n")
		("emitter-displacement", value<vector<string>>(), "string, which specifies where emitter should be placed inside the simulation\n"
										"\tto specify emitter position, emit area and emit velocity use next format: "
										"\"[\%f,\%f,\%f],\%f,[\%f,\%f,\%f],\%d\"\n")
		("viscosity", value<Real>()->default_value(1e-2), "fluid viscosity value\n")
		("rest-density", value<Real>()->default_value(1e+3), "fluid rest density\n")
		("border-rest-density", value<Real>()->default_value(3e+3), "border rest density\n")
		("friction", value<Real>()->default_value(1e-2), "fluid friction parameter\n")
		("iterations", value<size_t>()->default_value(3), "number iterations for position correction\n")
		("border-model-path", value<string>(), "path to obj file, where border model is resided\n")
		("sim-duration", value<Real>(), "simulation duration\n")
		("fluid-displacement", value<vector<string>>(), "specify lower and upper corner of fluid cube, that should be sampled for simulation\n"
											"\tplease use next format to define fluid cube: "
											"\"[\%f,\%f,\%f],[\%f,\%f,\%f]\"\n")
		("fluid-model", value<vector<string>>(), "path to mesh, that should be populated with fluid particles. !!!Mesh should be closed!!!\n")
		("sim-type", value<string>(), "simulation method type. Next variants are possible: PBF, WCF\n")
		("sample-dist", value<Real>(), "particle sampling distance\n")
		("pressure-stiffness", value<Real>()->default_value(80), "pressure coefficient for WCSPH\n")
		("eta", value<Real>()->default_value(1.2), "eta value - multiplier for compact support and smoothing length\n")
		("sim-name", value<string>()->default_value("new_simulation"), "simulation name."
														"\n\tAll vtk and cereal files will use simulation name prefix")
		("output-directory", value<string>()->default_value("./"), "specify output directory, where all cereal and vtk files will be saved\n")
		("clip-area", value<string>(), "specify area, outside of which fluid particles will be removed from the simulation\n")
		("max-velocity", value<Real>()->default_value(50.0f), "specify maximum velocity, which will be given to each particle (negative value means no velocity boundary)\n")
		("dbg", "debug option. If enabled then each physical update will be saved into vtk file\n")
		("sampling-type", value<string>()->default_value("hex"), "choose sampling type: hex for hexagonal, sqr - for standard square grid\n")
		("integration-scheme", value<string>()->default_value("smooth"), "choose simplectic Euler sheme: smooth, non-smooth\n")
		("border-sample-dist", value<Real>(), "specify distance of border particles\n")
		("pbf-velocity-multiplier", value<Real>()->default_value(1.0f), "specify velocity multiplier, which will be applied to particle after correction step in PBFSPH\n")
		("adhesion", value<Real>()->default_value(1.0f), "specify cohesion coefficient\n")
		("cohesion", value<Real>()->default_value(1.0f), "specify adhesion coefficient\n")
		("gravity", value<Real>()->default_value(-9.7), "environment gravity\n");

	//parse and assign command line options
	variables_map vm;
	store(parse_command_line(ac, av, po_def), vm);
	if(vm.count("help")){
		cout << po_def << endl;
		return 0;
	}
	validate_cmd_options(vm);
	assign_cmd_options(vm);
	cmdValues.print();

	//generate fluid particles
	FluidSystem fluid(cmdValues.fluid_rest_density, cmdValues.sampling_distance, cmdValues.eta);
	if(vm.count("fluid-displacement")){
		for(const string& fluid_disp : vm["fluid-displacement"].as<vector<string>>()){
			FluidDisplacement fd;
			parse_fluid_displacement(fluid_disp, fd);
			vector<Vector3R> positions;
			sample_fluid_cube(positions, fd.lower_corner, fd.upper_corner, cmdValues.sampling_distance);
			vector<Vector3R> velocity(positions.size(), Vector3R(0,0,0));
			fluid.add_fluid_particles(positions, velocity);
		}
	}
	if(vm.count("fluid-model")){
		for(const string& fluid_modle_path : vm["fluid-model"].as<vector<string>>()){
			vector<Vector3R> fluidParticles;
			sample_fluid_model(fluid_modle_path, 
								Vector3R(-5,-5,-5), 
								Vector3R(5,5,5),
								cmdValues.sampling_distance,
								fluidParticles);
			vector<Vector3R> velocity(fluidParticles.size(), Vector3R(0,0,0));
			fluid.add_fluid_particles(fluidParticles, velocity);
		}
	}
	fluid.setGravity(cmdValues.gravity);
	//add emmiters
	if(vm.count("emitter-displacement")) {
		for(const string& emit_disp : vm["emitter-displacement"].as<vector<string>>()){
			EmiterDisplacement ed;
			parse_emiter_displacement(emit_disp, ed);
			fluid.add_emitter(ed.max_particles, ed.em_pos, ed.em_area, ed.em_veloc);
		}
	}
	//generate border particles from border obj file
	vector<Vector3R> positions;
	sample_border_model_surface(positions, 
								Matrix4d::Identity(), 
								cmdValues.border_model_path, 
								cmdValues.border_sampling_distance, 
								cmdValues.samplingSheme);
	BorderSystem border(positions, 
						cmdValues.border_rest_density, 
						cmdValues.border_sampling_distance,
						cmdValues.eta);

	cout << "Number of fluid particles (except emited once): " << fluid.size() << endl;
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
	cout << "simulation finished. Total number of fluid particles: " << fluid.size() << endl;

	cout << endl;
	return 0;
}
