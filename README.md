# Avatar

This replication package contains the source code for surrogate-assisted model compression and training, as well as all trained compressed models.

## Environment configuration

To reproduce our experiments,  machines with GPUs and NVIDIA CUDA toolkit are required.

We provide a `Dockerfile` to help build the experimental environment. Please run the following scripts to to compile a docker image:
```
docker build -t YOUR_CUSTOM_TAG .
```
**NOTE⚠️:** Be careful with the torch version that you need to use. In our case, we use `torch==1.8.1+cu101`. Pls modify the `Dockerfile` according to your cuda version if needed.

Then, please run the docker container:
```
dokcer run -it -v YOUR_LOCAL_REPO_PATH:/root/Avatar --gpus all YOUR_CUSTOM_TAG
```

## How to run

Please use `docker attach` to go inside the docker container. 

Then, for each experiment in our paper, the scripts and instructions are in the `README.md` files under the each subfolder called `Avatar`. More specifically,

* Run the compressed CodeBERT on Clone Detection task: `Clone-Detection/BigCloneBench/CodeBERT/Avatar/README.md`
* Run the compressed GraphCodeBERT on Clone Detection task: `Clone-Detection/BigCloneBench/GraphCodeBERT/Avatar/README.md`
* Run the compressed CodeBERT on the Devign dataset of Vulnerability Detection task: `Vulnerability-Detection/Devign/CodeBERT/Avatar/README.md`
* Run the compressed GraphCodeBERT on the Devign dataset of Vulnerability Detection task: `Vulnerability-Detection/Devign/GraphCodeBERT/Avatar/README.md` 
* Run the compressed CodeBERT on the ReVeal dataset of Vulnerability Detection task: `Vulnerability-Detection/ReVeal/CodeBERT/README.md`
* Run the compressed GraphCodeBERT on the ReVeal dataset of Vulnerability Detection task: `Vulnerability-Detection/ReVeal/GraphCodeBERT/README.md`

We also release all trained 3 MB models and datasets in this repo. Users can directly run the scripts in the `README.md`.

## Misc

Due to the random nature of neural networks and our GA algorithm, users may obtain slightly different results. Please note that such results usually can be tolerated, i.e., they mostly do not conflict with the conclusions of the paper.

Here for anonymity, we remove all the open source `LICENSE` files.




