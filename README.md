# Strategic Inference in Stackelberg Games: Optimal Control for Revealing Adversary Intent

## Directory Structure
- `src/`: Contains the main code for the project.
- `src/core.py`: Core functionalities for the Stackelberg game. Contains classes and functions for defining the leader and follower parameters controls and dynamics, as well as game simulation and LSTM training. 
- `src/run_training.py`: Script to run the training process for the Stackelberg game. 
- `src/experiments.ipynb`: Jupyter notebook for running experiments and visualizing results.

## Usage 

1. **Install Dependencies**: Create a virtual environment using `conda` and install the required packages from `environment.yml`:

   ```bash
   conda env create -f environment.yml
   conda activate strategic-inference
   ```

2. **Run Training**: Execute the training script `run_training.py` to start the training process. 

3. **Run Experiments**: Open the `experiments.ipynb` notebook to visualize leader and follower controls, control comparisons, and single as well as multi period parameter estimation. 