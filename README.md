# Avatar

This replication package contains the source code for applying Avatar to optimize and train models. The obtained model checkpoints can be found at [this link](https://figshare.com/s/c674351fb51905f7e013).

## Environment configuration

To reproduce our experiments, machines with GPUs and NVIDIA CUDA toolkit are required.

We provide a `Dockerfile` to help build the experimental environment. Please run the following scripts to compile a docker image:
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

Then, for each experiment in our paper, the scripts and instructions are in the `README.md` files under each subfolder called `Avatar`.

## Misc

Due to the random nature of neural networks and our GA algorithm, users may obtain slightly different results. Please note that such results usually can be tolerated, i.e., they mostly do not conflict with the conclusions of the paper.

If you meet any problems when using our code, please contact Jieke SHI by [jiekeshi@smu.edu.sg](mailto:jiekeshi@smu.edu.sg). Many thanks!

This paper has been accepted by the Software Engineering in Society Track of the 46th IEEE/ACM International Conference on Software Engineering (ICSE '24). The arXiv version can be viewed by [this link](https://arxiv.org/abs/2309.04076).
If you use any code from our repo in your paper, pls cite:
```bibtex
@inproceedings{shi2024greening,
  author = {Shi, Jieke and Yang, Zhou and Kang, Hong Jin and Xu, Bowen and He, Junda and Lo, David},
  title = {Greening Large Language Models of Code},
  year = {2024},
  isbn = {9798400704994},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi-org.libproxy.smu.edu.sg/10.1145/3639475.3640097},
  doi = {10.1145/3639475.3640097},
  booktitle = {Proceedings of the 46th International Conference on Software Engineering: Software Engineering in Society},
  pages = {142–153},
  numpages = {12},
  keywords = {language models of code, configuration tuning, multi-objective optimization},
  location = {Lisbon, Portugal},
  series = {ICSE-SEIS'24}
}
```
