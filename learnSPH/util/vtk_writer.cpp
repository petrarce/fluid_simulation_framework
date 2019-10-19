#include <vtk_writer.h>

void learnSPH::saveParticlesToVTK(std::string path, const std::vector<Eigen::Vector3d>& particles, const std::vector<double>& particle_scalar_data, const std::vector<Eigen::Vector3d>& particle_vector_data)
{
	// Input checking
	if ((path.size() < 5) || (path.substr(path.size() - 4) != ".vtk")) {
		std::cout << "learnSPH error in saveParticlesToVTK: Filename does not end with '.vtk'" << std::endl;
		abort();
	}

	if ((particle_scalar_data.size() != 0) && (particles.size() != particle_scalar_data.size())) {
		std::cout << "learnSPH error in saveParticlesToVTK: Number of particles not equal to particle scalar data." << std::endl;
		abort(); 
	}
	if ((particle_vector_data.size() != 0) && (particles.size() != particle_vector_data.size())) {
		std::cout << "learnSPH error in saveParticlesToVTK: Number of particles not equal to particle vector data." << std::endl;
		abort(); 
	}

	// Open the file
	std::ofstream outfile(path, std::ios::binary);
	if (!outfile) {
		std::cout << "learnSPH error in saveParticlesToVTK: Cannot open the file " << path << std::endl;
		abort();
	}

	// Parameters
	int n_particles = (int)particles.size();

	// Header
	outfile << "# vtk DataFile Version 4.2\n";
	outfile << "\n";
	outfile << "BINARY\n";
	outfile << "DATASET UNSTRUCTURED_GRID\n";

	// Vertices
	{
		outfile << "POINTS " << n_particles << " double\n";
		std::vector<double> particles_to_write;
		particles_to_write.reserve(3 * n_particles);
		for (const Eigen::Vector3d& vertex : particles) {
			particles_to_write.push_back(vertex[0]);
			particles_to_write.push_back(vertex[1]);
			particles_to_write.push_back(vertex[2]);
		}
		swapBytesInplace<double>(&particles_to_write[0], (int)particles_to_write.size());
		outfile.write(reinterpret_cast<char*>(&particles_to_write[0]), particles_to_write.size() * sizeof(double));
		outfile << "\n";
	}

	// Connectivity
	{
		outfile << "CELLS " << n_particles << " " << 2 * n_particles << "\n";
		std::vector<int> connectivity_to_write;
		connectivity_to_write.reserve(2 * n_particles);
		for (int particle_i = 0; particle_i < (int)particles.size(); particle_i++) {
			connectivity_to_write.push_back(1);
			connectivity_to_write.push_back(particle_i);
		}
		swapBytesInplace<int>(&connectivity_to_write[0], (int)connectivity_to_write.size());
		outfile.write(reinterpret_cast<char*>(&connectivity_to_write[0]), connectivity_to_write.size() * sizeof(int));
		outfile << "\n";
	}

	// Cell types
	{
		outfile << "CELL_TYPES " << n_particles << "\n";
		int cell_type_swapped = 1;
		swapBytesInplace<int>(&cell_type_swapped, 1);
		std::vector<int> cell_type_arr(n_particles, cell_type_swapped);
		outfile.write(reinterpret_cast<char*>(&cell_type_arr[0]), cell_type_arr.size() * sizeof(int));
		outfile << "\n";
	}

	// Point data
	{
		int num_fields = 0;
		if (particle_scalar_data.size() > 0) { num_fields++; }
		if (particle_vector_data.size() > 0) { num_fields++; }

		outfile << "POINT_DATA " << n_particles << "\n";
		outfile << "FIELD FieldData " << std::to_string(num_fields) << "\n";


		if (particle_scalar_data.size() > 0) {
			outfile << "scalar" << " 1 " << n_particles << " double\n";
			std::vector<double> scalar_to_write;
			scalar_to_write.insert(scalar_to_write.end(), particle_scalar_data.begin(), particle_scalar_data.end());
			swapBytesInplace<double>(&scalar_to_write[0], (int)scalar_to_write.size());
			outfile.write(reinterpret_cast<char*>(&scalar_to_write[0]), scalar_to_write.size() * sizeof(double));
			outfile << "\n";
		}

		if (particle_vector_data.size() > 0) {
			outfile << "vector" << " 3 " << n_particles << " double\n";
			std::vector<double> vector_to_write;
			vector_to_write.reserve(3 * n_particles);
			for (const Eigen::Vector3d& vector : particle_vector_data) {
				vector_to_write.push_back(vector[0]);
				vector_to_write.push_back(vector[1]);
				vector_to_write.push_back(vector[2]);
			}
			swapBytesInplace<double>(&vector_to_write[0], (int)vector_to_write.size());
			outfile.write(reinterpret_cast<char*>(&vector_to_write[0]), vector_to_write.size() * sizeof(double));
			outfile << "\n";
		}
	}

	outfile.close();
}
