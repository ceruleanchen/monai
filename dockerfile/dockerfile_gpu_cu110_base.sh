#!/bin/bash

# Color
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'

# Absolute path to this script.
# e.g. /home/ubuntu/opencv_practice/dockerfile/dockerfile_opencv.sh
SCRIPT=$(readlink -f "$0")

# Absolute path this script is in.
# e.g. /home/ubuntu/opencv_practice/dockerfile
SCRIPT_PATH=$(dirname "$SCRIPT")

# Absolute path to the opencv path
# e.g. /home/ubuntu/opencv_practice
HOST_DIR_PATH=$(dirname "$SCRIPT_PATH")
echo "HOST_DIR_PATH  = "$HOST_DIR_PATH

# Host directory name
IFS='/' read -a array <<< "$HOST_DIR_PATH"
HOST_DIR_NAME="${array[-1]}"
echo "HOST_DIR_NAME  = "$HOST_DIR_NAME


if [ "$2" == "" ]
then
    VERSION="v1.0"
else
    VERSION=$2
fi
echo "VERSION        = "$VERSION

IMAGE_NAME="monai/base:$VERSION"
CONTAINER_NAME="monai_base_$VERSION"
echo "IMAGE_NAME     = "$IMAGE_NAME
echo "CONTAINER_NAME = "$CONTAINER_NAME

IFS=$'\n'
function Fun_EvalCmd()
{
    cmd_list=$1
    i=0
    for cmd in ${cmd_list[*]}
    do
        ((i+=1))
        printf "${GREEN}${cmd}${NC}\n"
        eval $cmd
    done
}


if [ "$1" == "build" ]
then
    lCmdList=(
                "docker build \
                    -f dockerfile_gpu_cu110_base.dockerfile \
                    -t $IMAGE_NAME ."
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "run" ]
then
    lCmdList=(
                "docker run --gpus all -itd \
                    --privileged \
                    --ipc=host \
                    --name $CONTAINER_NAME \
                    $IMAGE_NAME /bin/bash" \
                "docker exec -it $CONTAINER_NAME /bin/bash"
             )
                    # -p $HOST_API_PORT:8888 \
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "exec" ]
then
    lCmdList=(
                "docker exec -it $CONTAINER_NAME /bin/bash"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "start" ]
then
    lCmdList=(
                "docker start -ia $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "attach" ]
then
    lCmdList=(
                "docker attach $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "stop" ]
then
    lCmdList=(
                "docker stop $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "rm" ]
then
    lCmdList=(
                "docker rm $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "rmi" ]
then
    lCmdList=(
                "docker rmi $IMAGE_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

fi
