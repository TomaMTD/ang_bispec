python byspectrum.py -m cl -w all -l all -r 1 -N 1 -f 0
python byspectrum.py -m cl -w cl  -l all

python byspectrum.py -m bl -w all -l all -r 1 -N 1 -f 1
python byspectrum.py -m bl -w all -l noproj -r 1 -N 1 -f 1
python byspectrum.py -m bl -w all -l nolens -r 1 -N 1 -f 1
python byspectrum.py -m bl -w all -l all -r 1 -N 1 -fnl_local 1 -f 1

python byspectrum.py -m cl -w primordial -l all -r 0 -N 0 -f 0
python byspectrum.py -m bl -w primordial -l all -r 0 -N 0 -f 1 -fnl_local 1
