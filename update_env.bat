pipenv shell
pipenv update --dev
jupyter contrib nbextension install --user
jupyter nbextension enable toc2/main
jupyter nbextension varInspector/main
echo y | jupyter kernelspec uninstall casestudy1
python -m ipykernel install --user --name=casestudy1
