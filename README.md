# recurrent_name_generator
GAN based name generator: work in progress

This is an experiment to see if machines can learn to generate realistic sounding names.

## File details
### data
**male.txt**: contains male names list
**female.txt**: contains female names list
### codes
**data_utils.py**: contains data handling functions
**train.py**: the main training module
**generate.py**: contains function to generate names from already trained and saved models


## Get started:

To start training:
```shell
python train.py
```
To know more about options:
```shell
python train.py --help
```
