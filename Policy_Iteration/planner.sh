while [[ "$#" > 1 ]]; do case $1 in
    --algorithm) algo="$2";;
    --batchsize) batchsize="$2";;
    --randomseed) randomseed="$2";;
    --mdp) filepath="$2";;
    *) break;;
  esac; shift; shift
done

if [[ $algo == 'lp' || $algo == 'hpi' ]]
then
    randomseed=""
    batchsize=""
elif [[ $algo == 'bspi' ]]
then
    randomseed=""
elif [[ $algo == 'rpi' ]]
then
    batchsize=""
fi

chmod +x solver.py
cmd="python solver.py $filepath $algo $randomseed $batchsize"
#echo "$cmd"
$cmd

