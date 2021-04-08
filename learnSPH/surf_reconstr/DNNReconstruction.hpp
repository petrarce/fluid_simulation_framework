#include "NaiveMarchingCubes.hpp"
#include "ZhuBridsonReconstruction.hpp"
#include "SolenthilerReconstruction.hpp"
#include <fdeep/fdeep.hpp>
template<class BaseClass, class... Args>
class DNNReconstruction : public BaseClass
{
public:
	explicit DNNReconstruction(Args... args,
							   const std::string& dnnModelFile):
		BaseClass(args...),
		mModel(fdeep::load_model(dnnModelFile))
	{
	}
	DNNReconstruction(const DNNReconstruction& other):
		BaseClass(other)
	{}
private:
	struct Sample
	{
		std::vector<float> data;
		size_t baseCellIndex;
	};

	std::vector<Sample> generateSamples()
	{
		std::vector<Sample> samples;
		auto interCells = MarchingCubes::computeIntersectionVertices(0);
		while(!interCells.empty())
		{
			std::vector<float> sample;
			sample.reserve(125);
			MarchingCubes::CellIndex cI(*interCells.begin(), *this);
			assert(*cI != InvPrt);

			interCells.erase(interCells.begin());
			Eigen::Vector3li cell = MarchingCubes::cell(cI());
			for(int64_t i = -2; i <= 2; i++)
			{
				for(int64_t j = -2; j <= 2; j++)
				{
					for(int64_t k = -2; k <= 2; k++)
					{
						auto cellOffset = Eigen::Vector3li(i, j, k);
						Eigen::Vector3li nbCell = cell + cellOffset;
						MarchingCubes::CellIndex nbcI(MarchingCubes::cellIndex(nbCell), *this);
						while(*nbcI == InvPrt)
						{
							assert(nbcI() != cI());
							cellOffset /= 2;
							nbCell = cell + cellOffset;
							nbcI = MarchingCubes::cellIndex(nbCell);
						}
						interCells.erase(nbcI());
						sample.push_back(MarchingCubes::mMcVertexSdf[*nbcI]);
					}
				}
			}
			samples.push_back({std::move(sample), cI()});
		}
		return samples;
	}

	void correctLevelSet()
	{
		auto samples = generateSamples();
#pragma omp parallel for schedule(static)
		for(size_t i = 0; i < samples.size(); i++)
		{
			const Sample& sample = samples[i];
			MarchingCubes::CellIndex cI(sample.baseCellIndex, *this);
			auto baseCell = MarchingCubes::cell(cI());
			auto tensors = mModel.predict({fdeep::tensor(fdeep::tensor_shape(static_cast<size_t>(125)), sample.data)});
			assert(tensors.size() == 1);
			const auto& correctedSdf = *(tensors[0].as_vector());
			assert(correctedSdf.size() == 125);
			size_t currentIndex = 0;
			for(int64_t i = -2; i <= 2; i++)
			{
				for(int64_t j = -2; j <= 2; j++)
				{
					for(int64_t k = -2; k <= 2; k++)
					{
						auto nbC = baseCell + Eigen::Vector3li(i,j,k);
						MarchingCubes::CellIndex nbcI(MarchingCubes::cellIndex(nbC), *this);
						if(*nbcI == InvPrt)
						{
							currentIndex++;
							continue;
						}
						MarchingCubes::mMcVertexSdf[*nbcI] = correctedSdf[currentIndex];
						currentIndex++;
					}
				}
			}
		}

	}

	virtual void updateLevelSet()
	{
		BaseClass::updateLevelSet();
		correctLevelSet();
	}
private:
	const fdeep::model mModel;
};

typedef  DNNReconstruction<ZhuBridsonReconstruction, std::shared_ptr<learnSPH::FluidSystem> , const Eigen::Vector3d ,
								const Eigen::Vector3d , const Eigen::Vector3d , float > ZhuBridsonDNN;
typedef  DNNReconstruction<SolenthilerReconstruction, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d, const Eigen::Vector3d,
								const Eigen::Vector3d , float, float, float> SolenthilerDNN;
typedef  DNNReconstruction<NaiveMarchingCubes, std::shared_ptr<learnSPH::FluidSystem>, const Eigen::Vector3d,
								const Eigen::Vector3d, const Eigen::Vector3d, float> NaiveDNN;

