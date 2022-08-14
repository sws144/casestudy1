pipenv --rm
pipenv lock --keep-outdated
pipenv sync --dev
jupyter contrib nbextension install --user
jupyter nbextension enable toc2/main
jupyter nbextension enable varInspector/main
jupyter nbextension enable execute_time/ExecuteTime
echo y | jupyter kernelspec uninstall casestudy1
python -m ipykernel install --user --name=casestudy1
