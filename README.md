# TauRegression

python3 -m venv venv

source venv/bin/activate

pip install -r BasicNN/myenv_requirement.txt 

pip install jupyter

python -m ipykernel install --user --name=venv --display-name "Python (venv)"

Open GetAllRegressed_LBN_v3.ipynb in VS Code.
Click on the kernel selector (top right corner).
Select "Python (venv)" (or the environment you created).


## Second env for LBN keras 2 
python3 -m  venv env_keras2

source env_keras2/bin/activate


pip install -r /pbs/home/o/oponcet/private/TauRegression/env_keras2/myenv_keras2_requirement.txt 


pip install jupyter

python -m ipykernel install --user --name=env_keras2 --display-name "Python (env_keras2)"

Open MyGetAllRegressed_LBN.ipynb in VS Code.
Click on the kernel selector (top right corner).
Select "Python (venv)" (or the environment you created).

