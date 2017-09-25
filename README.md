# recurrent_name_generator
GAN based name generator: work in progress

This is an experiment to see if machines can learn to generate realistic sounding names, using generative adversarial networks. The implementation is in tensorflow and work is still in progress. It shows promising results and does NOT throw *absolute gibberish*. But it is not able to generate sufficient variation in the output as of now.

## Main concept
### Generator
Take a random noise and pass it repeatedly through a RNN to generate the name.
### Discriminator
Pass the generated names along with real names through a RNN to find whether the name is realistic or not.

## File details
### data
**male.txt**: contains male names list  
**female.txt**: contains female names list  
### codes
**data_utils.py**: contains data handling functions  
**train.py**: *the main training module*  
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
