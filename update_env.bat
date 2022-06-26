pipenv shell
pipenv update --dev
jupyter contrib nbextension install --user
jupyter kernelspec uninstall casestudy1
python -m ipykernel install --user --name=casestudy1
