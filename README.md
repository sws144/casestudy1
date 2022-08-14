# casestudy1

Case study example based on email

`casestudy.py` is the main script & associated notebook
<https://sws144-casestudy1-app-streamlit-fz26pv.streamlitapp.com/>

## Running jupyter notebook for analysis

1. `pipenv shell`, then `jupyter notebook`
1. backup is VSCode with jupyter interactive, `jupyter lab`

## MLFlow & Virtual Env Update

1. Install pipenv in desired environment ` pip install pipenv ` (`py -0p` to see which one is default) 
1. First time build/update:
    1. `pipenv shell`, then `update_env.bat` to install existing specified environment
    1. Backup: Terminal `pipenv sync --dev` to install env locally with piplock or 
    1. `pipenv update --dev` to **update based on Pipfile** and install environment
1. add specific package without updating rest if not necessary `pipenv install packagename --keep-outdated`
1. Terminal: `pipenv shell` to enter environment or in vs code, right click open terminal in folder with pipfile
1. `mlflow ui` to enter environment (omit --backend if want to see test runs)
1. To shut down, type "ctrl + c" in terminal
1. Optional: `mlflow gc` to clean up deleted runs (e.g. deleted from mlflow ui)

## Deploy

1. locally `streamlit run app_streamlit.app`
1. web <https://sws144-casestudy1-app-streamlit-fz26pv.streamlitapp.com/>