# Avatar

This replication package contains the source code for fine-tuning pre-trained models, surrogate-assisted model compression and training, as well as all trained compressed models.

## Environment configuration

We provide a Dockerfile to help build the experimental environment. Please run the following scripts to to compile a docker image:
```
docker build -t YOUR_CUSTOM_TAG .
```
Be careful with the torch version that you need to use, modify the Dockerfile according to your cuda version pls.

Then, please run the docker container:
```
dokcer run -it -v YOUR_LOCAL_REPO_PATH:/root/Avatar --gpus all YOUR_CUSTOM_TAG
```

## How to run

Please use `docker attach` to go inside the docker container. 

Then, for each experiment in our paper, the scripts and instructions are in the README.md files under each subfolder.

We also release all trained 3 MB models and datasets in this repo. 

## Misc

Here we remove all the open source `LICENSE` files for anonymity.
