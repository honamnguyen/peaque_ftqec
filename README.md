# PEAQUE - Fault-Tolerance Quantum Error Correction
- Fault-tolerance encoding verification: [ft_encoding_verification notebook](analysis/ft_encoding_verification.ipynb)


## Installation guide
- Create a `conda` environment `ENV_NAME`: 
```
conda create -n ENV_NAME python==3.9
conda activate ENV_NAME
```
- Install the required packages
```
pip install -r requirements.txt
sh setup.sh
```
- Attach the conda environment to jupyter
```
python -m ipykernel install --user --name=ENV_NAME
```
