all: pyinstall rinstall

pyinstall:
	pip install --user git+https://github.com/NKI-CCB/PRECISE

rinstall:
	Rscript install_R_dependencies.R
