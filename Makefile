#FUNCTION
define cecho
    @echo "\033[92m$(1)\033[0m"
endef

# rcnn-fer Docker Makefile
CPU_REGISTRY_URL=so77id
GPU_REGISTRY_URL=so77id
CPU_VERSION=latest
GPU_VERSION=latest
CPU_DOCKER_IMAGE=pytorch-opencv4-py3
GPU_DOCKER_IMAGE=pytorch-opencv4-py3
DOCKER_USER=mrodriguez
DOCKER_CONTAINER_NAME=mrodriguez


##############################################################################
############################# DOCKER VARS ####################################
##############################################################################
# COMMANDS
DOCKER_COMMAND=docker
NVIDIA_DOCKER_COMMAND=nvidia-docker


#HOST VARS
LOCALHOST_IP=127.0.0.1
HOST_TENSORBOARD_PORT=26006
HOST_NOTEBOOK_PORT=28888

#HOST CPU VARS
HOST_CPU_SOURCE_PATH=$(shell pwd)
HOST_CPU_DATASETS_PATH=/Users/mrodriguez/workspace/master/datasets
HOST_CPU_METADATA_PATH=/Users/mrodriguez/workspace/master/metadata

#HOST GPU PATHS
HOST_GPU_SOURCE_PATH=$(shell pwd)
HOST_GPU_DATASETS_PATH=/datasets/$(USER)
HOST_GPU_METADATA_PATH=/work/$(USER)/metadata/fer
HOST_GPU_DATASETS_SYN_PATH=/datasets/jalbarracin/syn/

#IMAGE VARS
IMAGE_TENSORBOARD_PORT=6006
IMAGE_NOTEBOOK_PORT=8888
IMAGE_SOURCE_PATH=/home/src
IMAGE_DATASETS_PATH=/home/datasets
IMAGE_DATASETS_SYN_PATH=/home/datasets/syn
IMAGE_METADATA_PATH=/home/metadata

# DOCKER vars
EXP_NAME=""
HOSTNAME=$(shell cat /etc/hostname)

# VOLUMES

CPU_DOCKER_VOLUMES = --volume=$(HOST_CPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
				     --volume=$(HOST_CPU_DATASETS_PATH):$(IMAGE_DATASETS_PATH) \
				     --volume=$(HOST_CPU_METADATA_PATH):$(IMAGE_METADATA_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH) \
				     --shm-size 8G

GPU_DOCKER_VOLUMES = --volume=$(HOST_GPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
				     --volume=$(HOST_GPU_DATASETS_PATH):$(IMAGE_DATASETS_PATH) \
 				     --volume=$(HOST_GPU_DATASETS_SYN_PATH):$(IMAGE_DATASETS_SYN_PATH) \
				     --volume=$(HOST_GPU_METADATA_PATH):$(IMAGE_METADATA_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH) \
				     --shm-size 8G


DOCKER_TENSORBOARD_PORTS = -p $(LOCALHOST_IP):$(HOST_TENSORBOARD_PORT):$(IMAGE_TENSORBOARD_PORT)
DOCKER_JUPYTER_PORTS = -p $(LOCALHOST_IP):$(HOST_NOTEBOOK_PORT):$(IMAGE_NOTEBOOK_PORT)

# IF GPU == false --> GPU is disabled
# IF GPU == true --> GPU is enabled
ifeq ($(GPU), true)
	DOCKER_RUN_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME) $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(GPU_VERSION)
	DOCKER_RUN_TENSORBOARD_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME) $(DOCKER_TENSORBOARD_PORTS) $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_JUPYTER_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME) $(DOCKER_JUPYTER_PORTS) $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(GPU_VERSION)
else
	DOCKER_RUN_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME)  $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_TENSORBOARD_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME)  $(DOCKER_TENSORBOARD_PORTS) $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_JUPYTER_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME)  $(DOCKER_JUPYTER_PORTS) $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
endif


# COMMANDS
JUPYTER_COMMAND=jupyter
TENSORBOARD_COMMAND=tensorboard
MKDIR_COMMAND=mkdir
WGET_COMMAND=wget

# URLs
C3D_URL=http://www.recod.ic.unicamp.br/~mrodriguez/weights/c3d.pickle

TENSORBOARD_PATH=$(IMAGE_METADATA_PATH)

setup s:
	@$(MKDIR_COMMAND) -p ./weigths
	@$(WGET_COMMAND) $(C3D_URL) -P ./weigths

run-test rtm: docker-print
	@$(DOCKER_RUN_COMMAND)


jupyter jp:
	$(call cecho, "[Jupyter] Running Jupyter lab")
	@$(EXPORT_COMMAND) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
	@$(JUPYTER_COMMAND) lab --ip=0.0.0.0 --allow-root

run-jupyter rj: docker-print
	@$(DOCKER_RUN_JUPYTER_COMMAND)  bash -c "make jupyter CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"; \
	status=$$?



tensorboard tb:
	$(call cecho, "[Tensorboard] Running Tensorboard")
	@$(TENSORBOARD_COMMAND) --logdir=$(TENSORBOARD_PATH) --host 0.0.0.0

run-tensorboard rt: docker-print
	@$(DOCKER_RUN_TENSORBOARD_COMMAND)  bash -c "make tensorboard TENSORBOARD_PATH=$(TENSORBOARD_PATH)"; \
	status=$$?


#PRIVATE
docker-print psd:
ifeq ($(GPU), true)
	$(call cecho, "[GPU Docker] Running gpu docker image...")
else
	$(call cecho, "[CPU Docker] Running cpu docker image...")
endif
