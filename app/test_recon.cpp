#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>
#include <experimental/filesystem>
#include <math.h>

#include <types.hpp>
//learnSPH
#include <learnSPH/core/vtk_writer.h>
#include <learnSPH/surf_reconstr/NaiveMarchingCubes.hpp>
#include <learnSPH/surf_reconstr/ZhuBridsonReconstruction.hpp>
#include <learnSPH/surf_reconstr/SolenthalerReconstruction.hpp>
#include <learnSPH/surf_reconstr/BlurredReconstruction.hpp>
#include <learnSPH/surf_reconstr/MlsReconstruction.hpp>
#include <learnSPH/surf_reconstr/MullerEtAlReconstruction.hpp>
#include <learnSPH/surf_reconstr/OnderikEtAlReconstruction.hpp>
//#include <learnSPH/surf_reconstr/MinDistReconstruction.hpp>
#include <learnSPH/core/storage.h>

#include <learnSPH/core/PerfStats.hpp>
//cereal
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

//boost
#include <boost/program_options.hpp>

//omp
#include <omp.h>

using namespace learnSPH;
using namespace std;
using namespace boost::program_options;

static void load_vectors(const std::string &path, std::vector<Vector3R> &data)
{
	std::ifstream is(path, std::ios::binary);

	cereal::BinaryInputArchive bia(is);

	size_t n_elems;

	bia(n_elems);

	data.reserve(n_elems);

	for (size_t i = 0; i < n_elems; i++) {

		Real x;
		Real y;
		Real z;

		bia(x, y, z);

		data.push_back(Vector3R(x, y, z));
	}
}

static void load_scalars(const std::string &path, std::vector<Real> &data)
{
	std::ifstream is(path, std::ios::binary);

	cereal::BinaryInputArchive bia(is);

	size_t n_elems;

	bia(n_elems);

	data.reserve(n_elems);

	for (size_t i = 0; i < n_elems; i++) {

		Real x;

		bia(x);

		data.push_back(x);
	}
}

static std::vector<std::string> filterPaths(const std::string& path, const std::string& pathPattern)
{
	std::vector<std::string> filteredPaths;
	std::regex pathRegex(pathPattern, std::regex::grep);
	for(const auto& p : std::experimental::filesystem::directory_iterator(path))
	{
		if(std::regex_search(p.path().string(), pathRegex))
			filteredPaths.push_back(p.path().string());
	}
	std::sort(filteredPaths.begin(), filteredPaths.end());
	return filteredPaths;
}

static void removeFugitives(vector<Vector3R>& positions, 
							vector<Real>& densities, 
							const Vector3R& lowerCorner, 
							const Vector3R upperCorner)
{
	assert(positions.size() == densities.size());
	for (size_t particleID = 0; particleID < positions.size();)
	{

		bool inside = true;

		inside &= (lowerCorner(0) <= positions[particleID](0));
		inside &= (lowerCorner(1) <= positions[particleID](1));
		inside &= (lowerCorner(2) <= positions[particleID](2));

		inside &= (positions[particleID](0) <= upperCorner(0));
		inside &= (positions[particleID](1) <= upperCorner(1));
		inside &= (positions[particleID](2) <= upperCorner(2));

		if (!inside)
		{
			positions[particleID] = positions.back();
			densities[particleID] = densities.back();
			positions.pop_back();
			densities.pop_back();
			continue;
		}
		particleID++;
	}


}

enum ReconstructionMethods
{
	NMC = 1,
	ZB,
	SLH,
	ZBBlur,
	NMCBlur,
	ZBMls,
	NMCMLS,
	MinDist,
	MullEtAll,
	OnderikEtAl,
	OnderikEtAlMls,
};

struct 
{
	Vector3R lowerCorner; //upper Domain corner
	Vector3R upperCorner; //lower Domain corner
	Real gridResolution;
	string simName;
	string simDir;
	Real initValue;
	ReconstructionMethods method;
	Real supportRad;
	Real tMin;
	Real tMax;
	Real sdfSmoothingFactor;
	size_t kernelSize;
	size_t kernelOffset;
	Real kernelDepth;
	bool blurSurfaceCellsOnly {false};
	size_t blurIterations {1};
	float colorFieldFactor {0};
	float similarityThreshold {0.5};
	size_t mlsMaxSamples {20};
	size_t mlsSamples {20};
	size_t mlsCurvatureParticles {20};
	float mlsSampleOverlapFactor {0.5};
	float mlsClusterFraction {0.5};
	
	void parse(const variables_map& vm)
	{
		if(vm.count("domain"))
		{
			auto values(arrayFromString(vm["domain"].as<string>()));
			if(values.size() != 6)
				throw invalid_argument("domain specified incorrecly. Reffer to help for the format");
			lowerCorner = Vector3R(values[0], values[1], values[2]);
			upperCorner = Vector3R(values[3], values[4], values[5]);
		} else
			throw invalid_argument("required option: --domain");
		
		if(vm.count("init-val"))
			initValue = vm["init-val"].as<Real>();
		else
			throw invalid_argument("required option: --domain");
		
		if(vm.count("grid-resolution"))
			gridResolution = vm["grid-resolution"].as<Real>();
		else
			throw invalid_argument("required option: --grid-resolution");
		
		if(vm.count("sim-name"))
			simName = vm["sim-name"].as<string>();
		else
			throw invalid_argument("required option: --sim-name");

		if(vm.count("sim-directory"))
			simDir = vm["sim-directory"].as<string>() + "/";
		else
			throw invalid_argument("required option: --sim-directory");
		
		if(vm.count("support-radius"))
			supportRad = vm["support-radius"].as<Real>();
		else
			throw invalid_argument("required option: --support-radius");
		if(vm.count("tmin"))
			tMin = vm["tmin"].as<Real>();
		else
			throw invalid_argument("required option: --tmin");
		
		if(vm.count("tmax"))
			tMax = vm["tmax"].as<Real>();
		else
			throw invalid_argument("required option: --tmax");

		if(vm.count("sdf-smoothing-factor"))
		{
			sdfSmoothingFactor = vm["sdf-smoothing-factor"].as<Real>();
			if(sdfSmoothingFactor < 0)
				throw invalid_argument("invalid --sdf-smoothing-factor. Should be in range [0, +inf]");
		}
		else
			throw invalid_argument("required option: --sdf-smoothing-factor");

		if(vm.count("blur-kernel-size"))
			kernelSize = vm["blur-kernel-size"].as<size_t>();
		else
			throw invalid_argument("required option: --blur-kernel-size");
		
		if(vm.count("blur-kernel-offset"))
			kernelOffset = vm["blur-kernel-offset"].as<size_t>();
		else
			throw invalid_argument("required option: --blur-kernel-offset");
		
		
		if(vm.count("blur-kernel-depth"))
		{
			kernelDepth = vm["blur-kernel-depth"].as<Real>();
			if(kernelDepth > 1 || kernelDepth< 0)
				throw invalid_argument("kernel depth should be between 0 and 1");
		}
		else
			throw invalid_argument("required option: --blur-kernel-depth");

		
		if(vm.count("method"))
		{
			string lmethod = vm["method"].as<string>();
			if(lmethod == "NaiveMC")
				method = ReconstructionMethods::NMC;
			else if(lmethod == "ZhuBridson")
				method = ReconstructionMethods::ZB;
			else if(lmethod == "Solenthaler")
				method = ReconstructionMethods::SLH;
			else if(lmethod == "ZhuBridsonBlurred")
				method = ReconstructionMethods::ZBBlur;
			else if(lmethod == "NaiveMCBlurred")
				method = ReconstructionMethods::NMCBlur;
			else if(lmethod == "ZhuBridsonMls")
				method = ReconstructionMethods::ZBMls;
			else if(lmethod == "NaiveMCMls")
				method = ReconstructionMethods::NMCMLS;
			else if(lmethod == "MinDist")
				method = ReconstructionMethods::MinDist;
			else if(lmethod == "MullEtAll")
				method = ReconstructionMethods::MullEtAll;
			else if(lmethod == "OnderikEtAl")
				method = ReconstructionMethods::OnderikEtAl;
			else if(lmethod == "OnderikEtAlMls")
				method = ReconstructionMethods::OnderikEtAlMls;
			else
				throw invalid_argument("unknown reconstruction method specified in  --method: " + lmethod);
		} else
			throw invalid_argument("required option: --method");

		if(vm.count("blur-surface-cells-only"))
			blurSurfaceCellsOnly = true;

		if(vm.count("blur-iterations"))
			blurIterations = vm["blur-iterations"].as<size_t>();
		else
			throw invalid_argument("required option: --blur-iterations");

		if(vm.count("cff"))
			colorFieldFactor = vm["cff"].as<float>();
		else
			throw invalid_argument("required option: --cff");

		if(vm.count("mls-max-samples"))
			mlsMaxSamples= vm["mls-max-samples"].as<size_t>();
		else
			throw invalid_argument("required option: --mls-max-samples");

		if(vm.count("mls-curvature-particles"))
			mlsCurvatureParticles= vm["mls-curvature-particles"].as<size_t>();
		else
			throw invalid_argument("required option: --mls-curvature-particles");

		if(vm.count("mls-sample-overlap-factor"))
			mlsSampleOverlapFactor= vm["mls-sample-overlap-factor"].as<float>();
		else
			throw invalid_argument("required option: --mls-sample-overlap-factor");

		if(vm.count("mls-samples"))
			mlsSamples = vm["mls-samples"].as<size_t>();
		else
			mlsSamples = mlsMaxSamples;

		if(vm.count("mls-cluster-fraction"))
		{
			mlsClusterFraction = vm["mls-cluster-fraction"].as<float>();
			if(mlsClusterFraction < 0 || mlsClusterFraction > 1)
				throw std::invalid_argument("mls-cluster-fraction should be between [0, 1]");
		}



	}
private:
	vector<Real> arrayFromString(const string& str)
	{
		vector<Real> res;
		string item;
		stringstream ss(str);
		Real val;
		while(getline(ss, item, ','))
		{
	
	
			pr_dbg("%s extracted", item.c_str());
			try{
				val = stod(item);}
			catch(...){
				continue;}
	
			res.push_back(val);
			pr_dbg("pushing %f to vector", val);
	
		}
		return res;
	}
} programInput;

int main(int argc, char** argv)
{
	globalPerfStats.startTimer("total execution time");
	options_description options;
	options.add_options()
			("help", "print this message")
			("domain", value<string>(), "set lower an upper corners of domain in format %f,%f,%f,%f,%f,%f")
			("init-val", value<Real>()->default_value(-0.5), "set initial value, that will be applied to discretisation grid ")
			("sim-name", value<string>(), "simulation name")
			("sim-directory", value<string>()->default_value("./"), "path to the simulation vtk files")
			("grid-resolution", value<Real>(), "uniform grid resolution for matching cubes")
			("method", value<string>()->default_value("NaiveMC"), "reconstruction type: NaiveMC[Blurred/Mls], ZhuBridson[Blurred/Mls], MinDist, Solenthaler")
			("support-radius", value<Real>()->default_value(2), "Support radius for position-based scalar fields")
			("tmin", value<Real>()->default_value(1), "lower bound for Solentiler evalue treshold")
			("tmax", value<Real>()->default_value(2), "upper bound for Solentiler evalue treshold")
			("sdf-smoothing-factor", value<Real>()->default_value(1), "scalar distance field smoothing factor")
			("blur-kernel-size", value<size_t>()->default_value(1), "kernel size for sdf bluring")
			("blur-kernel-offset", value<size_t>()->default_value(1), "bluring kernel offset")
			("blur-kernel-depth", value<Real>()->default_value(0.5), "depth of the bluring kernel in the direction normal to the gradient")
			("blur-surface-cells-only", "if flag is selected blurr will be applyed only on surface cells")
			("blur-iterations", value<size_t>()->default_value(1), "number of iterations blur is applied to the grid")
			("cff", value<float>()->default_value(0), "color field factor (for cff<0.05 color field particles detection is not applied)")
			("mls-samples", value<size_t>(), "number of mls samples, that should be examined in the neighborhood. Only mls-max-samples will be picked up after all from the investigated set randomly afterall. By default equal to mls-max-samples")
			("mls-max-samples", value<size_t>()->default_value(20), "maximum number of mls sample points")
			("mls-curvature-particles", value<size_t>()->default_value(20), "radius of flat surface in terms of fluid particles (diameter)")
			("mls-sample-overlap-factor", value<float>()->default_value(0.5), "factor of cluster size, a number of nearest neighbor samples in cluster that should be removed")
			("mls-cluster-fraction", value<float>()->default_value(0.5), "fraction of 0-level intersection cells, for which mls clusters will be generated")
			("parallel-frames", value<size_t>()->default_value(omp_get_max_threads()), "number of frames that should run in parallel")
			("tag", value<std::string>()->default_value(""), "addition simulation name tag for debug purposes")
			;
	variables_map vm;
	store(parse_command_line(argc, argv, options), vm);
	if(vm.count("help"))
	{
		cout << options << endl;
		return 0;
	}
	programInput.parse(vm);
	
	std::cout << "Per frame rendering running" << std::endl;

	std::vector<std::string> paramFiles = filterPaths(programInput.simDir, programInput.simName + "_params_[0-9]*.cereal");
	std::vector<std::string> positionFiles = filterPaths(programInput.simDir, programInput.simName + "_positions_[0-9]*.cereal");
	std::vector<std::string> densitiesFiles = filterPaths(programInput.simDir, programInput.simName + ".*_densities_[0-9]*.cereal");
	if(paramFiles.size() != positionFiles.size() || paramFiles.size() != densitiesFiles.size())
		throw runtime_error("lack of input files");

	pr_info("Max omp threads: %d", omp_get_max_threads());
    std::unique_ptr<MarchingCubes> mcbNew;
	string simtype;
	#pragma omp parallel for schedule(static, 1) private(mcbNew) num_threads(vm["parallel-frames"].as<size_t>())
	for (size_t t = 0; t < paramFiles.size(); t++) {
		if(!mcbNew)
		{
			switch(programInput.method)
			{
			case ReconstructionMethods::NMC :
				mcbNew = std::make_unique<NaiveMarchingCubes>(nullptr,
					programInput.lowerCorner, 
					programInput.upperCorner, 
					Vector3R(programInput.gridResolution, programInput.gridResolution, programInput.gridResolution), 
					programInput.initValue);
				simtype = string("NaiveMC")
						+ "_cff-" + to_string(programInput.colorFieldFactor)
						+ "_gr-" + to_string(programInput.gridResolution)
						+ "_iv-" + to_string(programInput.initValue);
				break;
			case ReconstructionMethods::MullEtAll :
				mcbNew = std::make_unique<MullerEtAlReconstruction>(nullptr,
					programInput.lowerCorner,
					programInput.upperCorner,
					Vector3R(programInput.gridResolution, programInput.gridResolution, programInput.gridResolution),
					programInput.initValue);
				simtype = string("MullerEtAlReconstruction")
						+ "_cff-" + to_string(programInput.colorFieldFactor)
						+ "_gr-" + to_string(programInput.gridResolution)
						+ "_iv-" + to_string(programInput.initValue);
				break;

			case ReconstructionMethods::ZB:
				mcbNew = std::make_unique<ZhuBridsonReconstruction>(nullptr,
					programInput.lowerCorner, 
					programInput.upperCorner, 
					Vector3R(programInput.gridResolution, programInput.gridResolution, programInput.gridResolution), 
					programInput.supportRad);
				simtype = string("ZhuBridson") + "_cff-" + to_string(programInput.colorFieldFactor) + "_gr-" + to_string(programInput.gridResolution) + "_sr-" + to_string(programInput.supportRad);
				break;
			case ReconstructionMethods::SLH:
				mcbNew = std::make_unique<SolenthalerReconstruction>(nullptr,
					programInput.lowerCorner, 
					programInput.upperCorner, 
					Vector3R(programInput.gridResolution, programInput.gridResolution, programInput.gridResolution), 
					programInput.supportRad,
					programInput.tMin,
					programInput.tMax);
				simtype = string("Solenthaler") + "_cff-" + to_string(programInput.colorFieldFactor) + "_gr-" + to_string(programInput.gridResolution) + "_sr-" + to_string(programInput.supportRad) +
						"_tmin-" + to_string(programInput.tMin) + "_tmax-" + to_string(programInput.tMax);
				break;
			case ReconstructionMethods::OnderikEtAl:
				mcbNew = std::make_unique<OnderikEtAlReconstruction>(nullptr,
					programInput.lowerCorner,
					programInput.upperCorner,
					Vector3R(programInput.gridResolution, programInput.gridResolution, programInput.gridResolution),
					programInput.supportRad,
					programInput.tMin,
					programInput.tMax);
				simtype = string("OnderikEtAl")
						+ "_cff-" + to_string(programInput.colorFieldFactor)
						+ "_gr-" + to_string(programInput.gridResolution)
						+ "_sr-" + to_string(programInput.supportRad)
						+ "_wmin-" + to_string(programInput.tMin)
						+ "_wmax-" + to_string(programInput.tMax);
				break;

			case ReconstructionMethods::ZBBlur:
				mcbNew = std::make_unique<ZhuBridsonBlurred>(nullptr,
					programInput.lowerCorner, 
					programInput.upperCorner, 
					Vector3R(programInput.gridResolution, programInput.gridResolution, programInput.gridResolution), 
					programInput.supportRad,
					programInput.sdfSmoothingFactor,
					programInput.kernelSize,
					programInput.kernelOffset,
					programInput.kernelDepth,
					programInput.blurSurfaceCellsOnly,
					programInput.blurIterations);
				simtype = string("ZhuBridsonBlurred") + "_cff-" + to_string(programInput.colorFieldFactor) + "_gr-" + to_string(programInput.gridResolution) + "_sr-" + to_string(programInput.supportRad) +
						"_sf-" + to_string(programInput.sdfSmoothingFactor) + "_ks-" + to_string(programInput.kernelSize) +
						"_ko-" + to_string(programInput.kernelOffset) + "_kd-" +
						to_string(programInput.kernelDepth) +
						((programInput.blurSurfaceCellsOnly)?"_sfco":"_nsfco") +
						"_bi-" + to_string(programInput.blurIterations);
				break;
			case ReconstructionMethods::NMCBlur:
				mcbNew = std::make_unique<NaiveBlurred>(nullptr,
					 programInput.lowerCorner,
					 programInput.upperCorner,
					 Vector3R(programInput.gridResolution,
							  programInput.gridResolution,
							  programInput.gridResolution),
					 programInput.initValue,
					 programInput.sdfSmoothingFactor,
					 programInput.kernelSize,
					 programInput.kernelOffset,
					 programInput.kernelDepth,
					 programInput.blurSurfaceCellsOnly,
					 programInput.blurIterations);
				simtype = string("NaiveMCBlurred") + "_cff-" + to_string(programInput.colorFieldFactor) + "_gr-" + to_string(programInput.gridResolution) + "_iv-" + to_string(programInput.initValue) +
						"_sf-" + to_string(programInput.sdfSmoothingFactor) + "_ks-" + to_string(programInput.kernelSize) +
						"_ko-" + to_string(programInput.kernelOffset) + "_kd-" +
						to_string(programInput.kernelDepth) +
						((programInput.blurSurfaceCellsOnly)?"_sfco":"_nsfco") +
						"_bi-" + to_string(programInput.blurIterations);
				break;
			case ReconstructionMethods::ZBMls:
				mcbNew = std::make_unique<ZhuBridsonMls>(nullptr,
														 programInput.lowerCorner,
														 programInput.upperCorner,
														 Vector3R(programInput.gridResolution, programInput.gridResolution, programInput.gridResolution),
														 programInput.supportRad,
														 programInput.sdfSmoothingFactor,
														 programInput.blurIterations,
														 programInput.mlsSamples,
														 programInput.mlsMaxSamples,
														 programInput.mlsCurvatureParticles,
														 programInput.mlsSampleOverlapFactor,
														 programInput.mlsClusterFraction);
#ifdef MLSV1
				simtype = string("ZhuBridsonMlsV1")
#else
#ifdef MLSV2
				simtype = string("ZhuBridsonMlsV2")
#else
				simtype = string("ZhuBridsonMlsV3")
#endif
#endif
						+ "_cff-" + to_string(programInput.colorFieldFactor)
						+ "_gr-" + to_string(programInput.gridResolution)
						+ "_sr-" + to_string(programInput.supportRad)
						+ "_sf-" + to_string(programInput.sdfSmoothingFactor)
						+ "_bi-" + to_string(programInput.blurIterations)
						+ "_maxs-" + to_string(programInput.mlsMaxSamples)
						+ "_mlss" + to_string(programInput.mlsSamples)
						+ "_cp-" + to_string(programInput.mlsCurvatureParticles)
						+ "_of-" + to_string(programInput.mlsSampleOverlapFactor)
						+ "_cf-" + to_string(programInput.mlsClusterFraction);



				break;
			case ReconstructionMethods::NMCMLS:
				mcbNew = std::make_unique<NaiveMls>(nullptr,
													programInput.lowerCorner,
													programInput.upperCorner,
													Vector3R(programInput.gridResolution, programInput.gridResolution, programInput.gridResolution),
													programInput.initValue,
													programInput.sdfSmoothingFactor,
													programInput.blurIterations,
													programInput.mlsSamples,
													programInput.mlsMaxSamples,
													programInput.mlsCurvatureParticles,
													programInput.mlsSampleOverlapFactor,
													programInput.mlsClusterFraction);
				simtype = string("NaiveMls")
						+ "_cff-" + to_string(programInput.colorFieldFactor)
						+ "_gr-" + to_string(programInput.gridResolution)
						+ "_iv-" + to_string(programInput.initValue)
						+ "_sf-" + to_string(programInput.sdfSmoothingFactor)
						+ "_bi-" + to_string(programInput.blurIterations)
						+ "_maxs-" + to_string(programInput.mlsMaxSamples)
						+ "_mlss" + to_string(programInput.mlsSamples)
						+ "_cp-" + to_string(programInput.mlsCurvatureParticles)
						+ "_of-" + to_string(programInput.mlsSampleOverlapFactor)
						+ "_cf-" + to_string(programInput.mlsClusterFraction);



				break;
			case ReconstructionMethods::OnderikEtAlMls:
				mcbNew = std::make_unique<OnderikMls>(nullptr,
													programInput.lowerCorner,
													programInput.upperCorner,
													Vector3R(programInput.gridResolution, programInput.gridResolution, programInput.gridResolution),
													programInput.supportRad,
													programInput.tMin,
													programInput.tMax,
													programInput.sdfSmoothingFactor,
													programInput.blurIterations,
													programInput.mlsSamples,
													programInput.mlsMaxSamples,
													programInput.mlsCurvatureParticles,
													programInput.mlsSampleOverlapFactor,
													programInput.mlsClusterFraction);
				simtype = string("OnderikEtAlMls")
						+ "_cff-" + to_string(programInput.colorFieldFactor)
						+ "_gr-" + to_string(programInput.gridResolution)
						+ "_sr-" + to_string(programInput.supportRad)
						+ "_wmin-" + to_string(programInput.tMin)
						+ "_wmax-" + to_string(programInput.tMax)
						+ "_sf-" + to_string(programInput.sdfSmoothingFactor)
						+ "_bi-" + to_string(programInput.blurIterations)
						+ "_maxs-" + to_string(programInput.mlsMaxSamples)
						+ "_mlss" + to_string(programInput.mlsSamples)
						+ "_cp-" + to_string(programInput.mlsCurvatureParticles)
						+ "_of-" + to_string(programInput.mlsSampleOverlapFactor)
						+ "_cf-" + to_string(programInput.mlsClusterFraction);
				break;

			default:
				throw std::runtime_error("unknown simulation type...");
			}
			simtype += "_domain-" + vm["domain"].as<string>() + vm["tag"].as<std::string>();
			mcbNew->setSimName(simtype);
		}
		vector<Real> params;
		vector<Vector3R> positions;
		vector<Real> densities;

		std::string filename = paramFiles[t];
		load_scalars(filename, params);
		filename = positionFiles[t];
		load_vectors(filename, positions);
		filename = densitiesFiles[t];
		load_scalars(filename, densities);

		removeFugitives(positions, densities, programInput.lowerCorner, programInput.upperCorner);
				
		vector<Vector3R> velocities(positions.size());
		shared_ptr<FluidSystem> fluidSystem = std::make_shared<FluidSystem>(
			std::move(positions), 
			std::move(velocities), 
			std::move(densities), 
			params[0]	/*restDensity*/,
			params[1]	/*compactSupport*/,
			params[2]	/*etaValue*/);
		std::regex integer("[0-9]+");
		std::smatch integerMatch;
		std::regex_search(filename, integerMatch, integer);
		if(!integerMatch.ready() || integerMatch.empty())
			throw std::invalid_argument("cereal file should have a sequence number");
		mcbNew->setColorFieldFactor(programInput.colorFieldFactor);
		mcbNew->setFrameNumber(integerMatch.str());
        vector<Vector3R> new_triangle_mesh((mcbNew->generateMesh(fluidSystem)));
		mcbNew->clearBuffers();
		assert((new_triangle_mesh.size() % 3) == 0);
		//generate and save triangular mesh
		vector<array<size_t, 3>> triangles;
		triangles.reserve(new_triangle_mesh.size() / 3);
		for(size_t i = 0; i < new_triangle_mesh.size(); i += 3) triangles.push_back({i, i + 1, i + 2});
		std::string surface_filename = programInput.simDir + programInput.simName + simtype + "_surface_" + integerMatch.str() + ".vtk";

		learnSPH::saveTriMeshToVTK(surface_filename, new_triangle_mesh, triangles);

		cout << "\nframe [" << integerMatch.str() << "] rendered" << endl;
		cout << "points in mesh: " << triangles.size() << std::endl;
	}
	globalPerfStats.stopTimer("total execution time");
	globalPerfStats.PrintStatistics(simtype);
}
