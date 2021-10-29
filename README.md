# smoking_hot_inverse_problems
First set up a python environment from environment.yml:

'''python
conda env create -f environment.yml
'''

To install required package Porepy:

git clone git@github.com:pmgbergen/porepy.git
cd porepy
/<location of your conda env>/bin/pip install -e .


If desired, install and setup ipykernel for your environment: 
(with environment activated)

conda install ipykernel
python -m ipykernel install --user --name <env_name> --display-name "Python (<env_name>)"
