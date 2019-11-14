#!/bin/bash

# setting arguments
sets=('rrinf_:-1_:2   '
      'dc4___:-1_:2   '
      'dc7___:-1_:2   '
      'rrinf_:-1_:+inf'
      'rrinf_:-1_:0   '
      'rrinf_:-1_:-1  '
      'rrinf_:-2_:2   ')

function subcheck {
    data=${1}
    prior=${2}
    lr=${3}

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

        for ((i = 0; i < 20; ++i)); do
            log="log_${data}_${prior}_${config}_g${geo}_a${alpha}_l${lr}_${i}.pt"
            path="logs/${data}_${prior}/${log}"
            if [ ! -f ${path} ]; then
                echo "missing ${log}"
            fi
        done
    done
}

for ln in n0; do
    # Part 1
    for pr in mmmk up; do
        subcheck emu_${ln} ${pr} 0
    done

    # Part 2
    for da in mm1k-small mm1k-large mmmmr; do
        for pr in mmmk up; do
            subcheck ${da}_${ln} ${pr} -2
        done
    done

    # Part 3
    subcheck lbwb_${ln} lbwb -2
done