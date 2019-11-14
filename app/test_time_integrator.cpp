//
// Created by nelson on 2019/11/5.
//
#include <stdlib.h>     // rand
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>    // std::max
#include <math.h>

#include <Eigen/Dense>
#include <data_set.h>
#include <types.hpp>
#include <particle_sampler.h>
#include <CompactNSearch>

#include <vtk_writer.h>
#include <solver.h>
#include <chrono>

using namespace CompactNSearch;
using namespace learnSPH;

int main(int argc, char** argv)
{
	assert(argc == 23);
	std::cout << "Welcome to the learnSPH framework!!" << std::endl;
	std::cout << "Generating test sample for Assignment 2...";

	Vector3R upper_corner_fluid = {stod(argv[1]),stod(argv[2]),stod(argv[3])};
	Vector3R lower_corner_fluid = {stod(argv[4]),stod(argv[5]),stod(argv[6])};
	Vector3R upper_corner_box = {stod(argv[7]),stod(argv[8]),stod(argv[9])};
	Vector3R lover_corner_box = {stod(argv[10]),stod(argv[11]),stod(argv[12])};
	Real sampling_distance = stod(argv[13]);
	Real compactSupportFactor = stod(argv[14]);
	Real preasureStiffness = stod(argv[15]);
	Real viscosity = stod(argv[16]);
	Real friction = stod(argv[17]);
	bool with_smoothing = stoi(argv[18]);
	bool withNavierStokes = stoi(argv[19]);
	Real defaultTimeStep = (stod(argv[20])); // time frame
	Real simulateDuration = (stod(argv[21])); // duration of the simulation
	string expName = argv[22]; // name of experience


    NormalPartDataSet* fluidParticles = sample_fluid_cube(upper_corner_fluid, lower_corner_fluid, 1000, sampling_distance);

    fluidParticles->setCompactSupportFactor(compactSupportFactor);
    NeighborhoodSearch ns(fluidParticles->getCompactSupport());

    auto fluidPointSet = ns.add_point_set((Real*)(fluidParticles->getParticlePositions().data()), fluidParticles->getNumberOfParticles(), true);

    cout << "Number of fluid particles: " << fluidParticles->getNumberOfParticles() << endl;

    BorderPartDataSet* borderParticles = sample_border_box(lover_corner_box, upper_corner_box, 3000, sampling_distance * 0.5, true);

    cout << "Number of border particles: " << borderParticles->getNumberOfParticles() << endl;

    ns.add_point_set((Real*)borderParticles->getParticlePositions().data(), borderParticles->getNumberOfParticles(), false);


    vector<vector<vector<unsigned int>>> particleNeighbors;
    particleNeighbors.resize(fluidParticles->getNumberOfParticles());

    const Vector3R gravity(0.0, -9.7, 0.0);
    vector<Vector3R>& particleForces = fluidParticles->getParticleForces();
    for(unsigned int i=0; i < particleForces.size(); i++){
        particleForces[i] = fluidParticles->getParticleMass() * gravity;
    }
//
    unsigned int nsamples = int(simulateDuration / defaultTimeStep);

    cout << "Duration: " << simulateDuration << endl;
    cout << "Default time step: "<< defaultTimeStep << endl;
    cout << "number of frames: "<< simulateDuration / defaultTimeStep << endl;

    for (unsigned int t = 0; t < nsamples; t++) {

        Real timeSimulation = 0;

        while (timeSimulation < 1) {
            ns.update_point_sets();
            for(int i = 0; i < fluidParticles->getNumberOfParticles(); i++){
                ns.find_neighbors(fluidPointSet, i, particleNeighbors[i]);
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
            //TODO set argument to customize velocity cap
            Real velocityCap = 300.0;
            Real vMaxNorm = 0;
            const Vector3R * fluidParticlesVelocity = fluidParticles->getParticleVelocitiesData();

            for (int iVelo=0; iVelo < fluidParticles->getNumberOfParticles(); iVelo++){
                if( (fluidParticlesVelocity[iVelo]).norm()>vMaxNorm){
                    vMaxNorm = (fluidParticlesVelocity[iVelo]).norm();
                }
            }
            vMaxNorm = min(vMaxNorm, velocityCap);

            Real delTimeCFL = 0.5*0.5*(fluidParticles->getParticleDiameter()/vMaxNorm);
            Real delTime;
            if (timeSimulation*defaultTimeStep + delTimeCFL >= defaultTimeStep){
                delTime = (1-timeSimulation)*defaultTimeStep;
                timeSimulation=1;
            }else{
                delTime = delTimeCFL;
                timeSimulation += delTime/defaultTimeStep;
            }

            if (not with_smoothing){
                learnSPH::Solver::semi_implicit_Euler(fluidParticlesAccelerations, *fluidParticles, delTime);
            }
            else{
                learnSPH::Solver::mod_semi_implicit_Euler(fluidParticlesAccelerations,
                                                          *fluidParticles,
                                                          particleNeighbors,
                                                          0.5,
                                                          delTime,
                                                          fluidParticles->getSmoothingLength());
            }

        }


        // Save
        std::string filename;

        vector<Vector3R>& particlePositions = fluidParticles->getParticlePositions();
        const Real maxRadius = 200.0; // To avoid exploded particles moves too far from center => view vanish
        bool beyondSphere = false;
        for (unsigned particleIndex = 0; particleIndex<fluidParticles->getNumberOfParticles(); particleIndex++){
            if (particlePositions[particleIndex].norm() > maxRadius){
                particlePositions[particleIndex] = 0.8*maxRadius*(particlePositions[particleIndex].normalized());
                beyondSphere = true;
            }
        }
        if (beyondSphere)
            std::cout<<"warning： particles away from center, frame: "<<t<<std::endl;
        filename = "res/assignment2/" + expName + '_' + std::to_string(t) + ".vtk";
        if (t%25==0)
            std::cout<<"epoch " + std::to_string(t)<<endl;

        learnSPH::saveParticlesToVTK(filename,
                                     fluidParticles->getParticlePositions(),
                                     fluidParticles->getParticleDencities(),
                                     fluidParticles->getParticleVelocities());
    }
    //save border
    std:string filename = "res/assignment2/border.vtk";
    vector<Vector3R> dummyVector(borderParticles->getNumberOfParticles());
    learnSPH::saveParticlesToVTK(filename,
                                 borderParticles->getParticlePositions(),
                                 borderParticles->getParticleVolume(),
                                 dummyVector);

    delete fluidParticles;
    std::cout << "completed!" << std::endl;
    std::cout << "The scene files have been saved in the folder `<build_folder>/res/assignment2/`. You can visualize them with Paraview." << std::endl;


    return 0;
}

