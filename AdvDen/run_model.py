from model import run_model
from params import getParams, saveParams
from plotting import plot_model

params = getParams()
run_model(params)
plot_model(params)
