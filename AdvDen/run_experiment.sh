export PYTHONPATH=/home/celdred/NewDEC:/home/celdred/NewDEC/external:/home/celdred/NewDEC/AdvDenMLP/src:/home/celdred/NewDEC/external/meshplex/src
rm *.png *.h5

python3 run_experiment.py experiment.cfg $1
