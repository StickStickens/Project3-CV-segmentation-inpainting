# CV-Project-3: semantic segmentation and image inpainting
Welcome to our project; We tried to train image segmentation model on ADE20k and then use
it's masks to remove certain objects from image by image inpainting model

## setup

after cloning the repo and going to repo folder do:
- On Linux/macOS: bash setup.sh
- On Windows: double-click setup.bat or run in Command Prompt

It is recommended to use virtual environment

To use DVC, you need to setup authorization for dagshub; first, create account on dagshub:
https://dagshub.com/
Then, you need to get your token from https://dagshub.com/user/settings/tokens
After that, run following commands:

dvc remote modify --local origin auth basic \
dvc remote modify --local origin user <YOUR_DAGSHUB_USERNAME> \
dvc remote modify --local origin password <YOUR_DAGSHUB_default_TOKEN>

After that, run dvc pull to get data and models

## File structure:

- src directory containing all project sourrce code, divide further into:
    - config, containing model/training specific parameters
    - dataset, containing dataset class and method to choose random sample from it
    - models, containing all models implementations (or gathering them via transforms library)
    - preprocessing, containing scripts to estimate dataset mean and std, run hiperparameter estimation
    and verifying model correctess (test if model overfits on small amount of data)
    - validation, containing scripts to see validation results as well as compare models between each other (both by metrics and visually)
- lama containing simple implementation of LaMa model
- models, containing trained (or fetched in case of lama) models
- data containing, well, data
- output folder, containing, if some function creates it, outputs produced by model validation , inpainting, or just some exemplary items from dataset
- global_params.json, describing most important parameters used almost everywhere
- project.ipynb containing most functionalities with their short descriptions

## How to run:
for model training, run:
python src/train.py 

adding following arguments: \ 
--model_class : Model class name to instantiate. \
--parameters_file : Path to the parameters JSON file. \
--save_model : Whether to save the trained model. \
--dataset_name : Name of the dataset class to use for training. \
--sample_number : Amount of samples used for training. (default is whole dataset)

For other functionalities, see project.ipynb







