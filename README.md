# PEAQUE - Fault-Tolerance Quantum Error Correction
- Fault-tolerance encoding verification: [ft_encoding_verification notebook](analysis/ft_encoding_verification.ipynb)
- Figures of merit: [figures_of_merit notebook](analysis/figures_of_merit.ipynb)

## Installation guide
- Clone this repository and change directory
```
git clone https://github.com/honamnguyen/peaque_ftqec.git
cd peaque_ftqec
```
- Create a `conda` environment `ENV_NAME`: 
```
conda create -n ENV_NAME python==3.10
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
- For example, run the following commands to install necessary packages to a `ftqec` environment:
```
git clone https://github.com/honamnguyen/peaque_ftqec.git
cd peaque_ftqec
conda create -n ftqec python==3.10
conda activate ftqec
pip install -r requirements.txt
sh setup.sh
python -m ipykernel install --user --name=ftqec
```