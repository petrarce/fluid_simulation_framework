//
// Created by nelson on 2019/11/23.
//
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
#include <marching_cubes.h>
#include <chrono>

using namespace CompactNSearch;
using namespace learnSPH;


int main(int argc, char** argv)
{
    assert(argc == 25);
    std::cout << "Welcome to the learnSPH framework!!" << std::endl;
    std::cout << "Generating test sample for Assignment 3.2...";

    Vector3R lower_corner_fluid = {stod(argv[1]),stod(argv[2]),stod(argv[3])};
    Vector3R upper_corner_fluid = {stod(argv[4]),stod(argv[5]),stod(argv[6])};

    Vector3R lower_corner_box = {stod(argv[7]),stod(argv[8]),stod(argv[9])};
    Vector3R upper_corner_box = {stod(argv[10]),stod(argv[11]),stod(argv[12])};

    auto box_center = (lower_corner_box + upper_corner_box) / 2.0;
    auto max_shift = (box_center - lower_corner_box).norm() * 1.2;

    Real sampling_distance = stod(argv[13]);
    Real eta = stod(argv[14]);
    Real stiffness = stod(argv[15]);
    Real viscosity = stod(argv[16]);
    Real friction = stod(argv[17]);
    bool do_velo_smooth = stoi(argv[18]);
    Real render_step = (stod(argv[19]));
    Real sim_duration = (stod(argv[20]));
    string sim_name = argv[21];
    Vector3R cubeResolution = {stod(argv[22]), stod(argv[23]), stod(argv[24])};

    FluidSystem* fluidParticles = sample_fluid_cube(lower_corner_fluid, upper_corner_fluid, 1000.0, sampling_distance, eta);

    cout << "Number of fluid particles: " << fluidParticles->size() << endl;

    BorderSystem* borderParticles = sample_border_box(lower_corner_box, upper_corner_box, 3000.0, sampling_distance * 0.5, eta * 0.5, true);

    cout << "Number of border particles: " << borderParticles->size() << endl;

    NeighborhoodSearch ns(fluidParticles->getCompactSupport());

    ns.add_point_set((Real*)(fluidParticles->getPositions().data()), fluidParticles->size(), true);

    ns.add_point_set((Real*)borderParticles->getPositions().data(), borderParticles->size(), false);

    vector<vector<vector<unsigned int> > > neighbors;

    neighbors.resize(fluidParticles->size());

    const Vector3R gravity(0.0, -9.7, 0.0);

    vector<Vector3R>& particleForces = fluidParticles->getExternalForces();

    MarchingCubes mcb;
    mcb.init(lower_corner_box, upper_corner_box, cubeResolution);
    vector<Vector3R> triangle_mesh;

    for(unsigned int i = 0; i < particleForces.size(); i++) particleForces[i] = fluidParticles->getMass() * gravity;

    unsigned int nsamples = int(sim_duration / render_step);

    cout << "Diameter: " << fluidParticles->getDiameter() << endl;
    cout << "Duration: " << sim_duration << endl;
    cout << "Default time step: "<< render_step << endl;
    cout << "number of frames: "<< sim_duration / render_step << endl;

    string filename = "res/assignment3/border.vtk";

    vector<Vector3R> dummyVector(borderParticles->size());

    learnSPH::saveParticlesToVTK(filename, borderParticles->getPositions(), borderParticles->getVolumes(), dummyVector);

    for (unsigned int t = 0; t < nsamples; t++) {

        Real timeSimulation = 0;

        int physical_steps = 0;

        while (timeSimulation < 1) {
            ns.update_point_sets();

            for(int i = 0; i < fluidParticles->size(); i++) ns.find_neighbors(0, i, neighbors[i]);

            learnSPH::calculate_dencities(fluidParticles, borderParticles, neighbors, fluidParticles->getSmoothingLength());


            vector<Vector3R> accelerations(fluidParticles->size(), gravity);

            learnSPH::calculate_acceleration(
                    accelerations,
                    fluidParticles,
                    borderParticles,
                    neighbors,
                    viscosity,
                    friction,
                    stiffness,
                    fluidParticles->getSmoothingLength());

            Real vMaxNorm = 0.0;

            vector<bool> &fluidActiveness = fluidParticles->getActiveness();

            vector<Vector3R> &fluidVelocities = fluidParticles->getVelocities();

            for (int i = 0; i < fluidParticles->size(); i++) if (fluidActiveness[i]) vMaxNorm = max(fluidVelocities[i].norm(), vMaxNorm);

            Real logic_step_upper_bound = 0.5 * (fluidParticles->getDiameter() / vMaxNorm);

//            cout << "\t| vMaxNorm: " << vMaxNorm << " | CFL-Step: " << logic_step_upper_bound << endl;

            Real logic_time_step;

            if (timeSimulation * render_step + logic_step_upper_bound >= render_step) {
                logic_time_step = (1 - timeSimulation) * render_step;
                timeSimulation = 1;
            } else {
                logic_time_step = logic_step_upper_bound;
                timeSimulation += logic_time_step / render_step;
            }

            if (!do_velo_smooth)
                learnSPH::symplectic_euler(accelerations, fluidParticles, logic_time_step);
            else
                learnSPH::smooth_symplectic_euler(accelerations, fluidParticles, neighbors, 0.5, logic_time_step, fluidParticles->getSmoothingLength());

            Real velocityCap = 50.0;

            vector<Vector3R> &fluidPositions = fluidParticles->getPositions();

            for (int i = 0; i < fluidParticles->size(); i++) {

                if (!fluidActiveness[i])
                    continue;

                if (fluidVelocities[i].norm() > velocityCap)
                    fluidVelocities[i] = velocityCap * fluidVelocities[i].normalized();

                if ((fluidPositions[i] - box_center).norm() > max_shift)
                    fluidActiveness[i] = false;
            }
            physical_steps++;
        }
        cout << "\n[" << physical_steps << "] physical updates were carried out for rendering frame [" << t << "]" << endl;

        string filename = "res/assignment3/" + sim_name + '_' + std::to_string(t) + ".vtk";

        learnSPH::saveParticlesToVTK(filename, fluidParticles->getPositions(), fluidParticles->getDensities(), fluidParticles->getVelocities());

        GeneralShape fluidObject( fluidParticles, 0.6, lower_corner_box, upper_corner_box, cubeResolution);
        mcb.setObject(&fluidObject);

        triangle_mesh.clear();
        mcb.getTriangleMesh(triangle_mesh);

        vector<array<int, 3>> triangles;
        for(int i = 0; i < triangle_mesh.size(); i += 3) triangles.push_back({i, i + 1, i + 2});

        std::string surface_filename = "res/assignment3/" + sim_name + "_surface" + std::to_string(t) + ".vtk";
        learnSPH::saveTriMeshToVTK(surface_filename, triangle_mesh, triangles);



    }
    delete fluidParticles;
    std::cout << "completed!" << std::endl;
    std::cout << "The scene files have been saved in the folder `<build_folder>/res/assignment2/`. You can visualize them with Paraview." << std::endl;

    return 0;
}

