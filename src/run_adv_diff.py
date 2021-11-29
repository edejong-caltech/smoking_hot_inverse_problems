import numpy as np
import scipy.sparse as sps
import sys
sys.path.append("..")
import porepy as pp
from PIL import Image
import glob
import os
import shutil
from porepy.fracs.meshing import grid_list_to_grid_bucket
import warnings

def main_test():
    n = 20
    num_faces = 2*n**2 + 2*n
    num_cells = n**2
    S = np.array([1.0])
    S_loc = np.array([[0.5, 0.5]])
    sense_t = np.array([1, 2])
    sense_loc = np.array([[0.2, 0.2],[0.2,0.6],[0.6,0.2],[0.6,0.2]])

    # modify if velocity field/diff_coeff are different than default
    velocity_field = -np.ones(num_faces)
    diff_coeff = np.ones(num_cells)
    run_toy_model(n, S, S_loc, sense_t, sense_loc, create_gif=False)
    print("success")

def run_toy_model(n, source_strength, source_locations, sensor_times, 
                  sensor_locations, velocity_field=None, diff_coeff=None,
                  create_gif=False, save_every=1, gif_name="adv_diff.gif"):
    '''
    OUTPUT: prints vectors of the local concentration at each of sensor_locations,
    and for every t in sensor_times

    INPUTS:
    n: number of grid points (n x n grid)
    source_strength: scalar array for emission sources
    source_locations: 2D vector array for emission locations, scaled for 1x1 domain
    sensor_times: scalar array for reporting outputs
    sensor_locations: 2D vector array for reporting outputs (locations)
    velocity_field (optional): should be as np.array(g.num_faces = 2*n^2 + 2*n)
    diff_coeff (optional): np.array(g.num_faces = 2*n^2 + 2*n)
    create_gif (optional): generate .png and export simulation as a gif
    save_every: frequency of images for optional gif
    gif_name: location/filename of optional gif
    '''

    t_max = sensor_times[-1]
    if create_gif:
        if not os.path.exists('tmp'): 
            os.makedirs('tmp')

    # create the grid
    g = pp.CartGrid([n,n])
    gb = grid_list_to_grid_bucket([[g]])
    g = gb.get_grids()[0]
    d = gb.node_props(g)
    kw_t = 'transport'

    # Transport related parameters
    if velocity_field is None:
        velocity_field = - np.ones(g.num_faces) 
    if diff_coeff is None:
        diff_coeff = 1.0 * np.ones(g.num_cells)
    assert len(velocity_field) == g.num_faces
    assert len(diff_coeff) == g.num_cells
    assert len(source_locations) == len(source_strength)
    add_transport_data(n, g, d, kw_t, velocity_field, diff_coeff, source_strength,
                        source_locations,t_max)

    # assemble transport problem
    grid_variable = "tracer"

    # Identifier of the discretization operator on each grid
    advection_term = "advection"
    source_term = "source"
    mass_term = "mass"
    diffusion_term = 'diffusion'

    # Discretization objects
    node_discretization = pp.Upwind(kw_t)
    source_discretization = pp.ScalarSource(kw_t)
    mass_discretization = pp.MassMatrix(kw_t)
    diffusion_discretization = pp.Tpfa(kw_t)

    # Assign primary variables on this grid. It has one degree of freedom per cell.
    d[pp.PRIMARY_VARIABLES] = {grid_variable: {"cells": 1, "faces": 0}}
    # Assign discretization operator for the variable.
    d[pp.DISCRETIZATION] = {
        grid_variable: {
            advection_term: node_discretization,
            source_term: source_discretization,
            mass_term: mass_discretization,
            diffusion_term: diffusion_discretization,
        }
    }
    assembler = pp.Assembler(gb)
    # Discretize all terms
    assembler.discretize()

    # Assemble the linear system, using the information stored in the GridBucket
    A, b = assembler.assemble_matrix_rhs()
    tracer_sol = sps.linalg.spsolve(A, b)
    assembler.distribute_variable(tracer_sol)

    # Use a filter to let the assembler consider grid and mortar variable only
    filt = pp.assembler_filters.ListFilter(variable_list=[grid_variable])
    assembler = pp.Assembler(gb)
    assembler.discretize(filt=filt)
    A, b = assembler.assemble_matrix_rhs(
        filt=filt, add_matrices=False
    )
    mass_term += "_" + grid_variable
    advection_term += "_" + grid_variable
    source_term += "_" + grid_variable
    diffusion_term += "_" + grid_variable
    time_step = d[pp.PARAMETERS][kw_t]["time_step"]
    t_max = d[pp.PARAMETERS][kw_t]["t_max"]
    lhs = A[mass_term] + time_step * (
        A[advection_term] + A[diffusion_term]
    )
    rhs_source_adv = b[source_term] + time_step * (
        b[advection_term] + b[diffusion_term]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        IEsolver = sps.linalg.factorized(lhs)

    # Initial condition
    tracer = np.zeros(rhs_source_adv.size)
    assembler.distribute_variable(
        tracer, variable_names=[grid_variable]
    )

    # find sensor location indices
    sensor_x = (np.round(sensor_locations[:,0]*n)*n + np.round(sensor_locations[:,1]*n)).astype(int)
    # set up and run
    n_steps = int(np.round(t_max / time_step))
    i_sensor_times = 0
    for i in range(n_steps):
        # export time step
        if i > 0 and i_sensor_times < len(sensor_times):
            if np.isclose(i % sensor_times[i_sensor_times], 0):
                i_sensor_times += 1
                assembler.distribute_variable(
                    tracer,
                    variable_names=[grid_variable],
                )
                local_state = d[pp.STATE][grid_variable]
                print(local_state[sensor_x])
                #exporter.write_vtu(export_fields, time_step=int(i // save_every))
                #pp.save_img("tracer_vis/tracer_" + str(int(i // save_every)) + ".png", gb, grid_variable, figsize=(15,12))
        if create_gif:
            pp.save_img("tmp/tracer_" + str(int(i // save_every)) + ".png", gb, grid_variable, figsize=(15,12))
        tracer = IEsolver(A[mass_term] * tracer + rhs_source_adv)

    if create_gif:
        frames = []
        imgs = glob.glob("tmp/*.png")
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)

        frames[0].save(gif_name, format='GIF',append_images=frames[1:],save_all=True,duration=300,loop=0)
        shutil.rmtree('tmp')


def add_transport_data(n, g, d, parameter_keyword, velocity_field, diff_coeff, 
                        source_strength, source_locations, tmax):
    # Method to assign data.
    # Boundary conditions: zero Dirichlet everywhere
    # TODO: Update boundary conditions
    #       Could be 0 dirichlet for large enough domain, or 0 Neumann at outflow boundaries and 0 dirichlet at inflow
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bc_val = np.zeros(g.num_faces)
    labels = np.array(["dir"] * b_faces.size)
    bc = pp.BoundaryCondition(g, b_faces, labels)

    # velocity field (on faces)
    flux_vals = velocity_field

    # diffusion coefficient
    diffusion = diff_coeff
    diff = pp.SecondOrderTensor(diffusion)

    # source terms
    f = np.zeros(g.num_cells)
    for (i, s) in enumerate(source_strength):
        loc = source_locations[i,:]
        f[int(loc[0]*n*n) + int(loc[1]*n)] = s

    # Inherit the aperture assigned for the flow problem
    specified_parameters = {
        "bc": bc,
        "bc_values": bc_val,
        "time_step": 1 / 2,
        "mass_weight": 1.0,
        "t_max": tmax,
        "darcy_flux": flux_vals,
        "second_order_tensor": diff,
        "source": f
        }
    pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)

#main_test()