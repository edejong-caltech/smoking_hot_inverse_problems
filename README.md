# smoking_hot_inverse_problems
The forward model takes a list of source strengths and locations for smoke
emissions, and prints out the local concentration at a specified list of
sensor times and locations. The forward model is contained in ```run_adv_diff.py```.

A sample notebook is included which sets up the parameters and calls the forward
model: ```run_forward_model.ipynb```.

For more detailed information on the porepy model and for development testing,
the notebook ```ex_adv_diff.ipynb``` should be helpful.



We have provided an environment.yml:
```
conda env create -f environment.yml
```

To install required package Porepy:
```
git clone https://github.com/pmgbergen/porepy.git
cd porepy
/<location of your conda env>/bin/pip install -e .
```

If desired, install and setup ipykernel for your environment: 
(with environment activated)
```
conda install ipykernel
python -m ipykernel install --user --name <env_name> --display-name "Python (<env_name>)"
```
