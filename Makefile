# Check if GPU is available - if `nvidia-smi` works then use GPUs
GPUS := $(shell command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1 && echo "--gpus all" || echo "")

# Set flag for docker run command
BASE_FLAGS=-it --rm
RUN_FLAGS=$(GPUS) $(BASE_FLAGS) -p 8889:8888

DOCKER_IMAGE_NAME = mava
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --tag $(IMAGE) .

run:
	$(DOCKER_RUN) python $(example)

bash:
	$(DOCKER_RUN) bash

lab:
	$(DOCKER_RUN)  jupyter lab --allow-root --NotebookApp.token='token' --no-browser --ip=0.0.0.0