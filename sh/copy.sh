#!/bin/bash

# setting arguments
sets=('rrinf_:-1_:2   '
      'dc4___:-1_:2   '
      'dc7___:-1_:2   '
      'rrinf_:-1_:0   '
      'rrinf_:-1_:-1  '
      'rrinf_:-2_:2   ')
      # // 'rrinf_:-1_:+inf')

function subcopy {
    data=${1}
    prior=${2}
    lr=${3}
    type=${4}

    root="logs/${data}_${prior}"
    mkdir -p ${root}
    rm -rf ${root}/${type}*

    for args in "${sets[@]}"; do
        buf=${args}
        config=${buf%%_:*}
        buf=${buf#*_:}
        geo=${buf%%_:*}
        buf=${buf#*_:}
        alpha=${buf}

        config=${config%%_*}
        geo=${geo%%_*}
        alpha=${alpha%% *}

        log="${type}_${data}_${prior}_${config}_g${geo}_a${alpha}_l${lr}_*.pt"
        path="${root}/${log}"
        echo -e "\033[32;1m${log}\033[0m"
        scp gao462@ml01.cs.purdue.edu:~/InftySGD/${path} ${root}
        if [ $? -ne 0 ]; then
            echo -e "\033[31;1mfail to scp ${path}\033[0m"
            exit
        fi
    done
}

for ln in n0; do
# //    # Part 0
# //    subcopy emu_${ln}   up    0 best2
# //    subcopy mm1k_${ln}  mmmk -1 best2
# //    subcopy mmul_${ln}  mmul -2 best2
# //    subcopy mmmmr_${ln} mmmk -1 best2
# //
# //    # Part 1
# //    subcopy emu_${ln} mmmk 0 log
# //    subcopy emu_${ln} up   0 log
# //
# //    # Part 2
# //    subcopy mm1k_${ln} mmmk -1 log
# //
# //    # Part 3
# //    subcopy mmul_${ln} mmul -2 log
# //
# //    # Part 4
# //    subcopy mmmmr_${ln} mmmk -1 log
done