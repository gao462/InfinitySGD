#!/bin/bash

# share parameters
seed=$(cat sh/seed | tail -1)
loop=20

# clean process buffer
host=${1}
host=${host%%.*}
proc=${host}.proc
rm -f ${proc}

# // # get device
# // if [ $# -gt 1 ]; then
# //     device=${2}
# // else
# //     device=-1
# // fi

# terminal arguments
terms=('ml01_:emu___:mmmk_:1____:100'
       'ml02_:emu___:up___:1____:100'
       'ml03_:mm1k__:mmmk_:0.1__:50 '
       'ml04_:mm1k__:up___:0.1__:50 '
       'ml05_:mmul__:mmul_:0.01_:50 '
       'ml06_:mmul__:up___:0.01_:50 '
       'ml07_:mmmmr_:mmmk_:0.1__:50 '
       'ml08_:mmmmr_:up___:0.1__:50 ')
# // terms=('ml01_:emu________:mmmk_:1____:100'
# //        'ml02_:emu________:up___:1____:100'
# //        'ml03_:mm1k-small_:mmmk_:0.1__:50 '
# //        'ml04_:mm1k-small_:up___:0.1__:50 '
# //        'ml05_:mm1k-large_:mmmk_:0.1__:50 '
# //        'ml06_:mm1k-large_:up___:0.1__:50 '
# //        'ml07_:mmmmr______:mmmk_:0.1__:150'
# //        'ml08_:mmmmr______:up___:0.1__:150'
# //        'ml09_:mmul-small_:mmul_:0.1__:50 '
# //        'ml10_:mmul-small_:up___:0.1__:50 '
# //        'ml11_:mmul-large_:mmul_:0.1__:50 '
# //        'ml12_:mmul-large_:up___:0.1__:50 ')

# try to hit host
flag=0
for args in "${terms[@]}"; do
    # split terminal argument segments
    buf=${args}
    mach=${buf%%_:*}
    buf=${buf#*_:}
    data=${buf%%_:*}
    buf=${buf#*_:}
    prior=${buf%%_:*}
    buf=${buf#*_:}
    lr=${buf%%_:*}
    buf=${buf#*_:}
    epoch=${buf}

    # clean terminal arguments
    mach=${mach%%_*}
    data=${data%%_*}
    prior=${prior%%_*}
    lr=${lr%%_*}
    epoch=${epoch%% *}

    # hit
    if [[ ${mach} == ${host} ]]; then
        flag=1
        break
    fi
done

# must hit a host
if [[ ${flag} -le 0 ]]; then
    echo "wrong host [${host}]"
    exit
fi

# wrapper
function run {
    echo ${1}
    ${1}
}

# setting arguments
sets=('rrinf_:0.1___:100'
      'dc4___:0.1___:100'
      'dc7___:0.1___:100'
      'rrinf_:0.1___:1  '
      'rrinf_:0.1___:0.1'
      'rrinf_:0.01__:100'
      'rrinf_:0.1___:inf')

# run all experiments
for args in "${sets[@]}"; do
    # split setting argument segments
    buf=${args}
    config=${buf%%_:*}
    buf=${buf#*_:}
    geo=${buf%%_:*}
    buf=${buf#*_:}
    alpha=${buf}

    # clean setting arguments
    config=${config%%_*}
    geo=${geo%%_*}
    alpha=${alpha%% *}

    # get exponent
    g=$(python -c "import math; print(int(math.log10(${geo})))")
    if [[ ${alpha} == inf ]]; then
        a='+inf'
    else
        a=$(python -c "import math; print(int(math.log10(${alpha})))")
    fi
    l=$(python -c "import math; print(int(math.log10(${lr})))")

    # construct log
    for ((i = 0; i < ${loop}; ++i)); do
        log="log_${data}_n0_${prior}_${config}_g${g}_a${a}_l${l}_${i}.pt"
        log="logs/${data}_n0_${prior}/${log}"
        if [ ! -f ${log} ]; then
            break
        fi
    done

    # check restart
    if [[ ${i} -eq ${loop} ]]; then
        continue
    else
        if [[ ${i} -ne 0 ]]; then
            i=$((i - 1))
        fi
        nsd=$((seed + i))
        nlp=$((loop - i))
    fi

    # run with all arguments
    cmd="python run.py --dataset ${data} --struct ${prior} --config ${config} --geo ${geo} --alpha ${alpha} --seed ${nsd} --lr ${lr} --epoch ${epoch} --loop ${nlp} --simulate --from ${i}"

    # // # put to device
    # // if [ ${device} -ne -1 ]; then
    # //     cmd="${cmd} --device ${device}"
    # // fi

    # run command
    run "${cmd}"

    # flag process
    printf "data=[%-10s] prior=[%-4s] config=[%-5s] geo=[%-4s] alpha=[%-3s] lr=[%-4s]\n" \
        ${data} ${prior} ${config} ${geo} ${alpha} ${lr} >> ${proc}
done