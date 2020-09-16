#include <vtk-9.0/vtkParticleReader.h>
#include <vtk-9.0/vtkSmartPointer.h>
#include <vtk-9.0/vtkUnstructuredGridReader.h>
#include <vtk-9.0/vtkUnstructuredGrid.h>
#include <vtk-9.0/vtkPointSet.h>
#include <vtk-9.0/vtkDataArray.h>
#include <vtk-9.0/vtkPoints.h>
#include <learnSPH/core/cereal_writer.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <cassert>
#include <learnSPH/core/kernel.h>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>

using namespace boost::program_options;
using namespace boost;

struct {
	Real particleRadius;
	std::string simulationName;
	std::vector<std::string> files;
	
	void parse(const variables_map& vm)
	{
		if(vm.count("pradius"))
			particleRadius = vm["pradius"].as<Real>();
		else
			throw std::runtime_error("required --pradius option");
		
		if(vm.count("sim-name"))
			simulationName = vm["sim-name"].as<std::string>();
		else
			throw std::runtime_error("required --sim-name option");
		
		if(vm.count("file"))
			files = vm["file"].as<std::vector<std::string>>();
		else
			throw std::runtime_error("required --file option");
	}
} appInputParams;
int main(int argc, char** argv)
{
	positional_options_description pos_opt;
	pos_opt.add("file", -1);
	
	options_description opt;
	opt.add_options()
			("help", "print this message")
			("pradius", value<Real>(), "particle radius")
			("sim-name", value<std::string>(), "simulation name")
			("file", value<std::vector<std::string>>(), "input files");
	
	variables_map vm;
	boost::program_options::store(command_line_parser(argc, argv).options(opt).positional(pos_opt).run(), 
		vm);
	if(vm.count("help"))
	{
		opt.print(std::cerr);
		return 0;
	}
	appInputParams.parse(vm);
	
			
	std::string simName = appInputParams.simulationName;
	Real partDiameter = appInputParams.particleRadius * 2;
	#pragma omp parallel for schedule(static, 1)
	for(size_t i = 0; i < appInputParams.files.size(); i++)
	{
		std::string& inpFile = appInputParams.files[i];
		assert(inpFile.find(".vtk") != string::npos);
		vtkSmartPointer<vtkUnstructuredGridReader> partReader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
		partReader->SetFileName(inpFile.c_str());
		partReader->Update();
		vtkPoints* dataArray = partReader->GetOutput()->GetPoints();
		std::vector<Eigen::Vector3d> pointVector(dataArray->GetNumberOfPoints(), Eigen::Vector3d::Zero());
		for(size_t j = 0; j < pointVector.size(); j++)
		{
			double pt[3];dataArray->GetPoint(j, pt);
			pointVector[j] = Vector3R(pt[0], pt[1], pt[2]);
		}
		
		save_vectors(simName + "_positions_" + to_string(i) + ".cereal", pointVector);
		vector<Real> dencities(pointVector.size(), 0);
		save_scalars(simName + "_densities_" + to_string(i) + ".cereal", dencities);
		vector<Real> params = { 2.4 * partDiameter, 
								1.2 * partDiameter, 
								1/6 * learnSPH::kernel::PI * partDiameter * partDiameter * partDiameter, 
								1000};
		save_scalars(simName + "_params_" + to_string(i) + ".cereal", params);
	}
	return 0;
}