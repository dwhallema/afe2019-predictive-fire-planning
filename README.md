# Random Forest prediction of potential fire control locations with scikit-learn (presented at the 8th International Fire Ecology and Management Congress, November 18-20, 2019, Tucson, Arizona) 

Author: [Dennis W. Hallema](https://www.linkedin.com/in/dennishallema) 

Description: Random Forest prediction of potential fire control locations (PCLs) for the 2018 Polecreek Fire in Utah. Prediction of PCLs is key to effective pre-fire planning and fire operations management. 

Depends: See `environment.yml`. 

Disclaimer: Use at your own risk. The authors cannot assure the reliability or suitability of these materials for a particular purpose. The act of distribution shall not constitute any such warranty, and no responsibility is assumed for a user's application of these materials or related materials.  

---

## Cloning this repository

1. Clone this repository onto your machine: 
   `git clone https://github.com/dwhallema/<repo>`, replace `<repo>` with the name of this repository. 
   This creates a new directory "repo" containing the repository files.
2. Install Anaconda and Python  
3. Install GDAL: `conda install gdal` (Important: do not install GDAL within the environment.)
4. Create a Python environment within the cloned directory using Anaconda: `conda env create`  
   This will download and install the dependencies listed in environment.yml.  
5. Activate the Python environment: `source activate <repo>` (or `conda activate <repo>` on Windows).  
6. Launch the Jupyter Notebook server: `jupyter notebook`  
7. In the opened browser tab, click on a ".ipynb" file to open it.  
8. To run a cell, select it with the mouse and click the "Run" button.  

Troubleshooting: 

* If you need to install dependencies manually run: `conda create -name <repo> dep`  
* If you need to update the Python environment run: `conda env update --file environment.yml`  
