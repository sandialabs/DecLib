from model import run_model
from params import getParams, saveParams
from plotting import plot_model
import subprocess
import glob
import sys
from simulations import get_simulations

params = getParams()

#change model, IC, nx, viscous coeff/type

experiment = sys.argv[2]

simulations = get_simulations(experiment)

for model,ic,nx,Nsteps,nout,dt,eps_type,eps_coeff,viscous_type in simulations:
    print("running " + model + " " + ic + " " + str(nx) + " " + viscous_type + " " + eps_type + " " + str(eps_coeff))
    params['init_cond'] = ic
    params['Nsteps'] = Nsteps
    params['nout'] = nout
    params['dt'] = dt
    params['nx'] = nx
    params['eps_coeff'] = eps_coeff
    params['eps_type'] = eps_type
    params['viscous_type'] = viscous_type

    params['model'] = model

    run_model(params)
    plot_model(params)
    saveParams(params, 'sim.cfg')
    dirname = model + '/' + ic + '/' + str(nx) + '/' + eps_type + '/' + 'eps=' + str(eps_coeff)
    subprocess.run(["mkdir", "-p", dirname])
    pngs = glob.glob("*.png")
    subprocess.run(["mv", *pngs, "sim.h5", "sim.cfg", dirname])
