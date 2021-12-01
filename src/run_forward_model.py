import numpy as np
import scipy.sparse as sps
import sys
sys.path.append("..")
import porepy as pp
from PIL import Image
from porepy.fracs.meshing import grid_list_to_grid_bucket
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt


""" Generate the configuration for a model run

Any named parameters passed to the function must correspond to
the names listed in `default_config`.
"""
def generate_model_config(**kwargs):
    default_config = {
        "Nx": 10,
        "Ny": 10,
        "widthx": 1.0,
        "widthy": 1.0,
        "time_step": 1.0,
        "tmax": 10.0,
        "source_strength": np.array([1.0]),
        "source_locations": np.array([[0.5, 0.5]]),
        "velocity_field": np.array([-1.0, -2.0]),  # [x-wind, y-wind]
        "diffusion_coefficent": 1.0,  # spatially uniform
        "sensor_times": np.array([2, 4, 6, 8]),  # times to save data at (physical time)
    }
    for key, value in kwargs.items():
        assert key in default_config, f"{key} is not a valid model config key."
        default_config[key] = value
    
    for key in ["source_strength", "source_locations", "velocity_field", "sensor_times"]:
        assert isinstance(default_config[key], np.ndarray), f"{key} is not a numpy array"
    
    gb = create_grid(default_config["Nx"], default_config["widthx"], default_config["Ny"], default_config["widthy"])
    default_config["gb"] = gb
    
    return default_config


""" Run the forward model with a given configuration

Parameters
----------
config          :: Model configuration (see `generate_model_config`)
save_output     :: Boolean. If true, save model state to file
create_gif      :: Optional. Generate .png and export simulation as a gif
gif_name        :: Filename of optional gif

Returns
-------
output_state    :: (Nt, Nx*Ny) array where Nt is the number of sensor save times and Nx*Ny is
                    the grid size.   
"""
def run_model(config,
    save_output=True,
    coarsegrain_output=False, coarse_grid=None,
    create_gif=False, gif_name="adv_diff.gif"
):
    # Create the grid
    gb = config["gb"]
    g = gb.get_grids()[0]
    d = gb.node_props(g)

    # Add data
    kw_t = 'transport'
    add_transport_data(config, g, d, kw_t)

    # Discretize the problem
    grid_variable = "tracer"
    assembler, IEsolver, mass_disc, rhs_source_adv = discretize_problem(gb, kw_t, grid_variable)

    # Set up and run
    time_step = config["time_step"]
    n_steps = int(np.round(config["tmax"] / time_step))

    source_locations = config["source_locations"]
    Nx, Ny = config["Nx"], config["Ny"]
    widthx, widthy = config["widthx"], config["widthy"]
    cell_widthx = widthx / Nx
    cell_widthy = widthy / Ny
    ij = f"{int(source_locations[0,0]/cell_widthx)}_{int(source_locations[0,1]/cell_widthy)}"
    # ij = f"{g.closest_cell(np.atleast_2d(source_locations[0,:]).T)[0]}"  # cell center indexing
    outpath = Path.cwd() / "data"
    outpath.mkdir(parents=True, exist_ok=True)
    output_filename = outpath / f"sample_data_{ij}.npy"
    sensor_times = config["sensor_times"]
    save_at = np.rint(sensor_times / time_step)  # time indices to save data at

    if create_gif:
        tempdir = Path(TemporaryDirectory().name)
        tempdir.mkdir(parents=True, exist_ok=True)

    output_state = np.zeros((len(sensor_times), g.num_cells))
    tracer = np.zeros(g.num_cells)
    j = 0
    for i in range(n_steps):
        # export time step
        if np.any(np.isclose(save_at, i)):
            assembler.distribute_variable(tracer, variable_names=[grid_variable])
            output_state[j, :] = np.copy(d[pp.STATE][grid_variable])
            j += 1
            # pp.save_img(f"tracer_vis/tracer_{i}.png", gb, grid_variable, figsize=(15,12))
            if create_gif:
                pp.save_img(str(tempdir / f"tracer_{j}.png"), gb, grid_variable, figsize=(15,12))
                plt.close()
        
        # Solve problem at current time step
        tracer = IEsolver(mass_disc * tracer + rhs_source_adv)

    output_state = output_state[:j, :]
    # If True, coarsegrain data to given resolution
    if coarsegrain_output:
        output_state = coarsegrain_data(coarse_grid, g, output_state)

    # Save output data
    if save_output:
        np.save(output_filename, output_state)

    if create_gif:
        frames = []
        imgs = list(tempdir.glob("*.png"))
        imgs.sort(key = lambda x: int(x.stem.split("_")[1]))
        for i in imgs:
           new_frame = Image.open(i)
           frames.append(new_frame)

        frames[0].save(outpath / gif_name, format="GIF", append_images=frames[1:], save_all=True, duration=300, loop=0)
    
    return output_state, gb


""" Create the grid

Parameters
----------
Nx      :: number of grid cells in the x direction
widthx  :: physical width of the grid in the x direction
Ny      :: If given, number of grid cells in the y direction. Otherwise same as Nx
widthy  :: If given, physical width of the grid in the y direction. Otherwise same as widthy

Returns
-------
gb      :: a PorePy grid bucket. Contains the grid and associated data.
""" 
def create_grid(Nx, widthx, Ny=None, widthy=None):
    # create the grid
    ny = Ny if Ny else Nx
    wy = widthy if widthy else widthx
    g = pp.CartGrid([Nx,ny], physdims=[widthx, wy])
    gb = grid_list_to_grid_bucket([[g]])
    return gb

def add_transport_data(config, g, d, parameter_keyword):
    # Method to assign data.
    # Boundary conditions: zero Dirichlet everywhere
    # TODO: Update boundary conditions
    #       Could be 0 dirichlet for large enough domain, or 0 Neumann at outflow boundaries and 0 dirichlet at inflow
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bc_val = np.zeros(g.num_faces)
    labels = np.array(["dir"] * b_faces.size)
    bc = pp.BoundaryCondition(g, b_faces, labels)

    # velocity field (on faces)
    xy_wind = config["velocity_field"]
    assert xy_wind.shape == (2,)  # [x-wind, y-wind]
    velocity_field = g.face_normals[:2,:].T @ xy_wind

    # diffusion coefficient
    diffusion = config["diffusion_coefficent"] * np.ones(g.num_cells)
    diff = pp.SecondOrderTensor(diffusion)

    # source terms
    source_strength = config["source_strength"]
    source_locations = config["source_locations"]
    assert source_locations.shape == (len(source_strength), 2)
    f = np.zeros(g.num_cells)
    source_inds = g.closest_cell(source_locations.T)
    for (ind, strength) in zip(source_inds, source_strength):
        f[ind] = strength

    # Inherit the aperture assigned for the flow problem
    specified_parameters = {
        "bc": bc,
        "bc_values": bc_val,
        "time_step": config["time_step"],
        "mass_weight": 1.0,
        "t_max": config["tmax"],
        "darcy_flux": velocity_field,
        "second_order_tensor": diff,
        "source": f
        }
    pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)


""" Discretize the problem"""
def discretize_problem(gb, parameter_keyword, grid_variable):
    g = gb.get_grids()[0]
    data = gb.node_props(g)

    # assemble transport problem

    # Identifier of the discretization operator on each grid
    advection_term = "advection"
    source_term = "source"
    mass_term = "mass"
    diffusion_term = "diffusion"

    # Discretization objects
    node_discretization = pp.Upwind(parameter_keyword)
    source_discretization = pp.ScalarSource(parameter_keyword)
    mass_discretization = pp.MassMatrix(parameter_keyword)
    diffusion_discretization = pp.Tpfa(parameter_keyword)

    # Assign primary variables on this grid. It has one degree of freedom per cell.
    data[pp.PRIMARY_VARIABLES] = {grid_variable: {"cells": 1, "faces": 0}}
    # Assign discretization operator for the variable.
    data[pp.DISCRETIZATION] = {
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
    # assembler = pp.Assembler(gb)
    assembler.discretize(filt=filt)
    A, b = assembler.assemble_matrix_rhs(
        filt=filt, add_matrices=False
    )
    mass_term += "_" + grid_variable
    advection_term += "_" + grid_variable
    source_term += "_" + grid_variable
    diffusion_term += "_" + grid_variable
    time_step = data[pp.PARAMETERS][parameter_keyword]["time_step"]
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
    tracer = np.zeros(g.num_cells)
    assembler.distribute_variable(
        tracer, variable_names=[grid_variable]
    )
    
    return assembler, IEsolver, A[mass_term], rhs_source_adv


""" Coarsegrain data from a fine to a coarse grid

It is assumed that every cell in the fine grid is fully contained
within some cell in the coarse grid.

Parameters
----------
g_coarse    :: Grid onto which data is coarse grained
g_fine      :: Grid on which `fine_data` exists
fine_data   :: 1D array of data to be coarse grained. 
                Must be ordered according to `g_fine` cell center ordering.

Returns
-------
coarse_data :: 1D array of coarsened data. Ordering is consistent with g_coarse
                cell center ordering.
"""
def coarsegrain_data(g_coarse, g_fine, fine_data):
    Nt = fine_data.shape[0]
    fine_coarse_map = g_coarse.closest_cell(g_fine.cell_centers)
    coarse_data = np.zeros((Nt, g_coarse.num_cells))
    for t in np.arange(Nt):  # iterate every time step
        for i in np.arange(g_coarse.num_cells):  # iterate every coarse cell
            coarse_data[t,i] = np.mean(fine_data[t,:][fine_coarse_map==i])
    return coarse_data
