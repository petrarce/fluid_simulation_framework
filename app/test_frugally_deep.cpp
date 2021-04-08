// main.cpp
#include <fdeep/fdeep.hpp>
#include <random>
int main(int argc, char** argv)
{
	const auto model = fdeep::load_model(argv[1]);
	std::vector<float> inp_data(125, 0);
	std::mt19937 gen(4);
	for(int i = 0; i < 125; i++)
		inp_data[i];
	const auto result = model.predict(
		{fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(125)),
		inp_data)});
	std::cout << fdeep::show_tensors(result) << std::endl;
}
