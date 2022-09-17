# Avatar

This replication package contains the source code for fine-tuning pre-trained models, model simplification and training, as well as all trained compressed models.

## Environment configuration

We provide a Dockerfile to help build the experimental environment. Please run the following scripts to to compile a docker image:
```
docker build -t YOUR_CUSTOM_TAG .
```
Be careful with the torch version that you need to use, modify the Dockerfile according to your cuda version pls.

Then, please run the docker:
```
dokcer run -it -v YOUR_LOCAL_REPO_PATH:/root/Compressor --gpus all YOUR_CUSTOM_TAG
```

After that, pls go inside the docker first, and then install some necessary libraries:

```
pip3 install -r requirements.txt
```

## How to run

The scripts and instructions for each experiment are in the `README.md` files under each subfolder.