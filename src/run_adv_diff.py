import numpy as np
from numpy.lib.function_base import diff
from numpy.lib.utils import source
import scipy.sparse as sps
import porepy as pp
from PIL import Image
import glob
from porepy.fracs.meshing import grid_list_to_grid_bucket


def main_test():
    n = 20
    num_faces = 2*n**2 + 2*n
    velocity_field = -np.ones(num_faces)
    diff_coeff = np.ones(num_faces)
    source_strength = 1.0
    run_toy_model(n, velocity_field, diff_coeff, source_strength)
    print("success")

def run_toy_model(n, velocity_field, diff_coeff, source_strength):
    '''
    n: number of grid points (n x n grid)
    velocity_field: shoud be as np.array(g.num_faces = 2*n^2 + 2*n)
    diff_coeff: np.array(g.num_faces = 2*n^2 + 2*n)
    source_strength: scalar value for emission source; located at center of domain
    '''
    # create the grid
    g = pp.CartGrid([n,n])
    gb = grid_list_to_grid_bucket([[g]])
    g = gb.get_grids()[0]
    d = gb.node_props(g)
    kw_t = 'transport'

    # Transport related parameters 
    assert len(velocity_field) == g.num_faces
    assert len(diff_coeff) == g.num_faces
    add_transport_data(g, d, kw_t, velocity_field, diff_coeff, source_strength)

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

    IEsolver = sps.linalg.factorized(lhs)


def add_transport_data(g, d, parameter_keyword, velocity_field, diff_coeff, source_strength):
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
    # Estimate diffusion coefficient of smoke particulates in air
    diffusion = diff_coeff
    diff = pp.SecondOrderTensor(diffusion)

    # source term: defaults at center of domain for now
    # TODO: possible location of emission and/or multiple emission sources
    f = np.zeros(g.num_cells)
    f[int(g.num_cells / 2) + 10] = source_strength

    # Inherit the aperture assigned for the flow problem
    specified_parameters = {
        "bc": bc,
        "bc_values": bc_val,
        "time_step": 1 / 2,
        "mass_weight": 1.0,
        "t_max": 10.0,
        "darcy_flux": flux_vals,
        "second_order_tensor": diff,
        "source": f
        }
    pp.initialize_default_data(g, d, parameter_keyword, specified_parameters)

main_test()