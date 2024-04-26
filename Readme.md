# Supplementary code for "Self-organized tissue mechanics underlie embryonic regulation" by Caldarelli et al. 

Copyright 2024 Alexander Chamolly


- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [License](#license)

## System Requirements

The code uses the FEniCS finite element package (https://fenicsproject.org) to solve for the flows in the embryo. The code used to generate the figures in the publication uses FEniCS 2019.1.0 which is now considered a legacy version. The two python notebooks `Fenics2019_Intact.ipynb` and `Fenics2019_Cut.ipynb` use this library. 

The notebook `Fenicsx0.7.1_Intact.ipynb` together with the libraries `lib_fem.py` and `lib_plot.py` contains a more future-proof re-implementation using the more recent FEniCSx v0.7.1 backend of the intact version of the embryo. The main motivation for this is that legacy FEniCS is not straightforward to install on Apple Silicon. 

In order to replicate the sinulations reported in the paper precisely we recommend using the legacy FEniCS version. We only include the re-implementation to aid those that do not have access to a machine with x86-compatible CPU.

In addition to FEniCS, the code depends on a number of standard python libraries which can be installed using `pip` or `conda`.

The code was tested on a 2016 Intel MacBook Pro running macOS 12.4 (legacy FEniCS) and a 2021 M1 Pro MacBook Pro running macOS 14.4.1 (FEniCSx) .

## Installation Guide

### Legacy FEniCS

To execute the code, it is necessary to perform a custom installation of legacy FEniCS together with some custom python libraries, which can be a bit cumbersome. The following is not necessarily the only way to do so, but worked on our test machine (with Intel CPU). The installation time is about 5-15 minutes.

First, you need to install Docker Desktop (https://www.docker.com).

Then download legacy FEniCS (https://fenicsproject.org/) using the console command 

	curl -s https://get.fenicsproject.org | bash

The fenicsproject website then instructs you to use the command 'fenicsproject run' in the console, but actually this is deprecated. Instead, use the following command (read on before executing):

	docker run -ti -p 127.0.0.1:8888:8888 -v /path/to/working/folder:/home/fenics/shared quay.io/fenicsproject/stable

Here `/path/to/working/folder` should be replaced by the path to your working directory. *It is essential for performance that this directory is empty!* If you have executed this command with the wrong path, destroy the container through the docker dashboard and start over!

This command creates a docker container inside of which the FEniCS python libraries will be available. In order to run the code, you will need to install the python shapely package inside of this container. To do so, enter into the console after running the command above:

	pip install shapely --user

The Docker application will allow you to open a dashboard, inside of which you should be able to see the container running that you created. When the container is deleted, the shapely installation will be gone. In order to avoid reinstalling it on launch every time, make sure to only stop the container from running (through the dashboard) but not to delete/destroy it. The container will have an auto-generated name (mine is called `festive_shirley`) which you will need to restart it, as detailed below.

#### Restarting container of legacy FEniCS

To restart the container, make sure docker is running, open a console and run the commands (individually, and in order)

	docker start festive_shirley
	docker exec -it festive_shirley bash -l 
	su fenics

where `festive_shirley` should be replaced by the name of your container.


#### Running FEniCS in a jupyter noteboook

Finally, to run FEniCS in a jupyter notebook, make sure you follow the instructions above and then create a jupyter session by running the command
	
	jupyter-notebook --ip=0.0.0.0

This command generates a link that you can copy and paste into your browser to open a Jupyter notebook with FEniCS. Your working directory will be in the shared/ folder. Open or create any jupyter notebook, and run the python code in there. As long as the file is in the shared/ folder, it will be saved on your hard drive if you save it inside of the container. However, your operating system will not be able to find any file that you place outside of the shared/ folder, and any files outside of the shared/ folder will be deleted when the container is destroyed.

### FEniCSx

For installation of the FEniCSx backend, please see the up to date instructions at https://fenicsproject.org. We recommend using conda instead of docker for much less headaches with any additional dependencies.

## Usage

Once installed, the code can be run by just executing the python notebooks. Depending on the speed of the CPU, a simulation should take around 5-30 minutes.

The 'Intact' embryo files allow for the simulation of a intact epiblast, and contains a `hair` flag that optionally enables a local friction term that models the hair. The geometrical extent of the hair can be modified by modifying the definition of the `hairmesh` variable in the meshing section. The 'Cut' notebook simulates ablated embryo halves and contains `reattach` and `antpost` flags that toggle between no-slip and no-stress boundary conditions on the cut, and simulation of the anterior/posterior halves respectively.

Various parameters are defined at the beginning of the files and can be edited. Since this code was not designed to be production code, it is possible that unforeseen bugs, in particular numerical instability, arise when these are modified significantly from the default values. Also, the code may contain leftover functionality that was used in earlier stages of the project but not used in the final publication, which may be buggy.

## Output

### Legacy FEniCS

Both notebooks define a `dataDir` variable which specifies the name of an output directory that will be created (or wiped!) when the code is executed. In this directory, text files with the values of variables defined on the margin are stored, together with the output of the finite elements in .pvd format for further processing, e.g. in Paraview. A copy of the notebook that was executed is also stored.

#### Demo

Example output of the notebook `Fenicsx0.7.1_Intact.ipynb` is provided in the `test_intact_2019' folder. On-the-fly visualisation of the simulation is preserved in the notebook file.

### FEniCSx

This version of the code will use .xdmf files to save the finite elements, and a pandas dataframe for the margin variables. Also, the code backup is not implemented.

## License

This code is covered under the **GNU General Public License v3.0**.
