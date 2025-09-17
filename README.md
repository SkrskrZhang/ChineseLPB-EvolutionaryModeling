# Usage
1. Install Python 3.7. For convenience, 
execute the following command.

   `pip install -r requirements.txt`

2. Generate Figs. 
Run `figs in manuscript.py` to get the national patterns and zonal differences 
in LPBs response to nutrient.

3. Generate Modeling Results. 
The Data used in Modeling are provided in [Google Drive](https://drive.google.com/drive/folders/1zNG8akvqXo5uwaStmMz1kIhmmkDEpfAE?usp=sharing). 
Download and place the model input data in the folder `./Data`.
Run `model_optimize.py` to get the optimized paras values and the calibrated model results. 
Run `model.py` to get the model scenario results.
