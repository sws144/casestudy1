# casestudy1

Case study example based on email

`casestudy.py` is the main script & associated notebook

## Running jupyter notebook for analysis

1. `pipenv shell`, then `jupyter notebook`
1. backup is VSCode with jupyter interactive, `jupyter lab`

## MLFlow & Virtual Env Update

1. Install pipenv in desired environment ` pip install pipenv ` (`py -0p` to see which one is default) 
1. First time build & update environment:
    1. `pipenv shell` (to enter virtual environment) and `update_env.bat` (windows commands)
    1. backup:  Terminal `pipenv sync --dev` to install env locally with piplock or `pipenv update --dev` to update and install environment
1. Terminal: `pipenv shell` to enter environment or in vs code, right click open terminal in folder with pipfile
1. `mlflow ui` to enter environment (omit --backend if want to see test runs)
1. To shut down, type "ctrl + c" in terminal
1. Optional: `mlflow gc` to clean up deleted runs (e.g. deleted from mlflow ui)
