//
// Created by nelson on 2019/11/5.
//
#include <stdlib.h>     // rand
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // std::max

#include <Eigen/Dense>
#include <data_set.h>
#include <types.hpp>
#include <particle_sampler.h>
#include <CompactNSearch>

#include <vtk_writer.h>
#include <solver.h>
#include <chrono>
using namespace CompactNSearch;


int main(int argc, char** argv)
{
    assert(argc == 15);
    std::cout << "Welcome to the learnSPH framework!!" << std::endl;
    std::cout << "Generating test sample for 3.a) Assignment 2...";

    Vector3R upper_corner = {stod(argv[1]),stod(argv[2]),stod(argv[3])};
    Vector3R lover_corner = {stod(argv[4]),stod(argv[5]),stod(argv[6])};
    Real sampling_distance = stod(argv[7]);
    Real compactSupportFactor = stod(argv[8]);
    Real preasureStiffness = stod(argv[9]);
    Real viscosity = stod(argv[10]);
    Real friction = stod(argv[11]);
    bool with_smoothing = stoi(argv[12]);
    bool withNavierStokes = stoi(argv[13]);
    size_t nsamples = stoi(argv[14]);


    NormalPartDataSet* fluidParticles =
            static_cast<NormalPartDataSet*>(learnSPH::ParticleSampler::sample_normal_particles(upper_corner,
                                                                                               lover_corner,
                                                                                               1000,
                                                                                               sampling_distance));

    fluidParticles->setCompactSupportFactor(compactSupportFactor);
    NeighborhoodSearch ns(fluidParticles->getCompactSupport());

    auto fluidPartilesPset = ns.add_point_set((Real*)(fluidParticles->getParticlePositions().data()),
                                              fluidParticles->getNumberOfParticles(),
                                              true,
                                              true,
                                              true);
    std::cout<< "number of fluid particles: " << fluidParticles->getNumberOfParticles() << endl;

//    vector<Vector3R> dummyVector;
//    BorderPartDataSet dummyBorderParticles(dummyVector, 1, 1);
//
//    ns.add_point_set((Real*)dummyBorderParticles.getParticlePositions().data(),
//                     dummyBorderParticles.getNumberOfParticles(),
//                     false, true, true);

    Vector3R distanceVector = upper_corner - lover_corner;
    upper_corner += distanceVector.normalized();
    lover_corner -= distanceVector.normalized();

    BorderPartDataSet* borderParticles =
            static_cast<BorderPartDataSet*>(learnSPH::ParticleSampler::sample_border_box(
                    lover_corner,
                    upper_corner,
                    1000, 
                    sampling_distance,
                    true));

    ns.add_point_set((Real*)borderParticles->getParticlePositions().data(),
                     borderParticles->getNumberOfParticles(),
                     false, true, true);


    Real unit_timeframe = 0.002;
    vector<vector<vector<unsigned int>>> particleNeighbors;
    particleNeighbors.resize(fluidParticles->getNumberOfParticles());

    const Vector3R gravity(0.0,-9.7,0.0);
    vector<Vector3R>& particleForces = fluidParticles->getParticleForces();
    for(unsigned int i=0; i<particleForces.size(); i++){
        particleForces[i] = fluidParticles->getParticleMass() * gravity;
    }
//
    for (unsigned int t = 0; t<nsamples; t++){

        ns.update_point_sets();
        for(int i = 0; i < fluidParticles->getNumberOfParticles(); i++){
            ns.find_neighbors(fluidPartilesPset, i, particleNeighbors[i]);
        }

        learnSPH::Solver::calculate_dencities(*fluidParticles,
                                              *borderParticles,
                                              particleNeighbors,
                                              fluidParticles->getSmoothingLength());

        // consider only gravity as external forces
        vector<Vector3R> fluidParticlesAccelerations(fluidParticles->getNumberOfParticles(), gravity);
        if(withNavierStokes){
            learnSPH::Solver::calculate_acceleration(fluidParticlesAccelerations,
                                                     *fluidParticles,
                                                     *borderParticles,
                                                     particleNeighbors,
                                                     viscosity, 
                                                     friction, 
                                                     preasureStiffness, 
                                                     fluidParticles->getSmoothingLength());
        }


        if (with_smoothing){
            learnSPH::Solver::semi_implicit_Euler(fluidParticlesAccelerations, *fluidParticles, unit_timeframe);
        }
        else{
            learnSPH::Solver::mod_semi_implicit_Euler(fluidParticlesAccelerations, 
                                                      *fluidParticles, 
                                                      particleNeighbors, 
                                                      0.5, 
                                                      unit_timeframe, 
                                                      fluidParticles->getSmoothingLength());
        }

        // Save
        std::string filename;
        if(with_smoothing)
            if(withNavierStokes)
                filename = "../res/2_3a/smooth_navier_stokes" + std::to_string(t) + ".vtk";
            else
                filename = "../res/2_3a/smooth_" + std::to_string(t) + ".vtk";
        else
            filename = "../res/2_3a/no_smooth_" + std::to_string(t) + ".vtk";
        std::cout<<"epoch " + std::to_string(t)<<endl;

        learnSPH::saveParticlesToVTK(filename,
                                     fluidParticles->getParticlePositions(),
                                     fluidParticles->getParticleDencities(),
                                     fluidParticles->getParticleVelocities());

    }
    //save border
    std:string filename = "../res/2_3a/border.vtk";
    vector<Vector3R> dummyVector(borderParticles->getNumberOfParticles());
    learnSPH::saveParticlesToVTK(filename,
                                 borderParticles->getParticlePositions(),
                                 borderParticles->getParticleVolume(),
                                 dummyVector);

//        for(Real density : fluidParticles->getParticleDencities()){
//            fprintf(stderr, "%f\n", density);
//        }



    delete fluidParticles;
    std::cout << "completed!" << std::endl;
    std::cout << "The scene files have been saved in the folder `<build_folder>/res/2_3a/`. You can visualize them with Paraview." << std::endl;


    return 0;
}

