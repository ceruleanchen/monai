#!/bin/bash
# Absolute path to this script file
SCRIPT_FILE=$(readlink -f "$0")

# Absolute directory this script is in
SCRIPT_DIR=$(dirname "$SCRIPT_FILE")

# Absolute path to the MONAI_DIR
MONAI_DIR=$(dirname "$SCRIPT_DIR")


# Get config.yaml
function Fun_ConvertConfigResult()
{
    config_result=$1
    print_result=$2

    config_result=`echo $config_result | sed 's/{//g' | sed 's/}//g' | sed 's/'\''//g'`
    IFS=",: " read -a config <<< $config_result

    for (( i=0; i<${#config[@]}; i+=2 ))
    do
        key=`echo ${config[$i]^^} | sed -e 's/\r//g'`
        value=`echo ${config[$i+1]} | sed -e 's/\r//g'`
        eval "$key=$value"
        if [ "$print_result" = "1" ]
        then
            printf "${YELLOW}%-20s = $value${NC}\n" $key
        fi
    done
}

IFS=$'\n'
function Fun_EvalCmd()
{
    cmd_list=$1
    i=0
    for cmd in ${cmd_list[*]}
    do
        ((i+=1))
        printf "${GREEN}\n${cmd}${NC}\n"
        eval $cmd
        exit_code=$?

        if [[ $exit_code = 0 ]]; then
            printf "${GREEN}[Success] ${cmd} ${NC}\n"
        else
            printf "${RED}[Failure: $exit_code] ${cmd} ${NC}\n"
            exit 1
        fi
    done
}


# # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Set production in global configuration        #
# # # # # # # # # # # # # # # # # # # # # # # # # # #
printf "${GREEN}cd $MONAI_DIR/config${NC}\n"
cd $MONAI_DIR/config

printf "${GREEN}python config.py --set_production inference${NC}\n"
config_result=`python config.py --set_production inference`
Fun_ConvertConfigResult "$config_result" 1

if [ "$PRODUCTION" != "retrain_aifs" ]
then
    # Color
    NC='\033[0m'
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
fi

# # # # # # # # # # # # # # # # #
#     Global Configuration      #
# # # # # # # # # # # # # # # # #
lCmdList=(
            "cd $MONAI_DIR/config" \
            "python config.py"
         )
Fun_EvalCmd "${lCmdList[*]}"


# # # # # # # # # # #
#     Training      #
# # # # # # # # # # #
lCmdList=(
            "cd $MONAI_DIR/src" \
            "python inference_api.py"
         )
Fun_EvalCmd "${lCmdList[*]}"
