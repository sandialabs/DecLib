from DecLib import PETSc
#import yaml
#import sys

from helpers import createMeshes
from initialconditions import getInitialCondition
from timesteppers import getTimestepper
from output import Output
from dynamics_mlp import AdvDensMLPStatistics, AdvDensMLPDiagnostics, AdvDensMLPDynamics
from dynamics_v import AdvDensVStatistics, AdvDensVDiagnostics, AdvDensVDynamics
from dynamics_advection import AdvectionStatistics, AdvectionDiagnostics, AdvectionDynamics
from dynamics_cons import AdvDensConsStatistics, AdvDensConsDiagnostics, AdvDensConsDynamics
from params import getParams

#FIX SRC LOCATION ISSUES HERE IF POSSIBLE

# from initial_conditions import InitialCondition
# from output import Output
# from timesteppers import Timestepper
# from advdens_dynamics import advdensdynamics1D, Statistics1D, Diagnostics1D
#
#cfgfilename = sys.argv[1]
#cfgfile =  open(cfgfilename + '.yaml", mode="r")
#params = yaml.safe_load(cfgfile)

#HOW DO WE ACTUALLY SET DEFAULTS HERE?
#DO WE REALLY NEED TO BE ABLE TO SET DEFAULTS?
#Probably, yes
#Create a parameters class
#Basically should just be an overloaded dictionary or set of dictionaries with defaults for various keys!


def run_model(params):
    # # create ic
    ic = getInitialCondition(params)

    #get meshes
    meshes = createMeshes(params)

    #EVENTUALLY EXPAND TO SUPPORT MULTICOMPONENT CE, MULTICOMPONENT ANELASTIC, ARBITRARY THERMO, ETC.
    if params['model'] == 'mlp':
        dyn = AdvDensMLPDynamics(meshes, params, ic)
        diag = AdvDensMLPDiagnostics(dyn)
        stats = AdvDensMLPStatistics(dyn, params['compute_bnd_fluxes'])
    if params['model'] == 'v':
        dyn = AdvDensVDynamics(meshes, params, ic)
        diag = AdvDensVDiagnostics(dyn)
        stats = AdvDensVStatistics(dyn, params['compute_bnd_fluxes'])
    if params['model'] == 'advection':
        dyn = AdvectionDynamics(meshes, params, ic)
        diag = AdvectionDiagnostics(dyn)
        stats = AdvectionStatistics(dyn, params['compute_bnd_fluxes'])
    if params['model'] == 'cons':
        dyn = AdvDensConsDynamics(meshes, params, ic)
        diag = AdvDensConsDiagnostics(dyn)
        stats = AdvDensConsStatistics(dyn, params['compute_bnd_fluxes'])

    dyn.set_IC()

    ts = getTimestepper(params['timestepper'], dyn, stats, params['compute_bnd_fluxes'])
    out = Output('sim', dyn, diag, stats)

    print('computing initial diagnostics/stats + output')
    t = 0.0
    diag.compute(t)
    stats.compute(0, t)
    out.output_const()
    out.output(0)
    print('starting time loop')


    for k in range(1,params['Nsteps']+1):
        #print(k)
        ts.take_step(k, params['dt'], t)
        #stats before output so stats correctly output
        if (k%params['nstat'] == 0):
            #print('stat')
            stats.compute(k//params['nstat'], t)
        if (k%params['nout'] == 0):
            print(k, 'output')
            diag.compute(t)
            out.output(k//params['nout'])
        t = t + params['dt']
