#!/bin/bash

PWD=`pwd`
port=5002
horizon=10000
nRuns=100
hostname="localhost"
banditFile="$PWD/data/instance-5.txt"

algorithms=('rr' 'epsilon-greedy' 'UCB' 'KL-UCB' 'Thompson-Sampling')
# algorithm="rr"
algorithm=${algorithms[4]}
# Allowed values for algorithm parameter(case-sensitive)
# 1. epsilon-greedy
# 2. UCB
# 3. KL-UCB
# 4. Thompson-Sampling
# 5. rr

epsilon=0.1

SERVERDIR=./server
CLIENTDIR=./client

OUTPUTFILE=$PWD/serverlog.txt

# randomSeed=0

banditFile="$PWD/data/$1"
horizon=$2
algorithm=$3
randomSeed=$4
epsilon=$5
# OUTPUTFILE="${PWD}/log/serverlog${randomSeed}.txt"
numArms=$(wc -l $banditFile | cut -d" " -f1 | xargs)

pushd $SERVERDIR
cmd="./startserver.sh $numArms $horizon $port $banditFile $randomSeed $OUTPUTFILE &"
#echo $cmd
$cmd > /dev/null
popd

sleep 1

pushd $CLIENTDIR
cmd="./startclient.sh $numArms $horizon $hostname $port $randomSeed $algorithm $epsilon&"
#echo $cmd
$cmd #> /dev/null
popd

