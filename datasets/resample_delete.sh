# ffmpeg required.
# this script resamples all wav files in same/sub directory into sample rate 22050Hz, and then removes the original.
# copy this to root directory of data and ./resample_delete.sh (you may need to chmod a+x.)
# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    printf '%.3d' $? >&3
    )&
}
resample_delete(){
    echo $1
    ffmpeg -loglevel panic -i "$1" -ar 22050 "${1%.*}-22k.wav"
    rm $1
}

N=16
open_sem $N
for f in $(find . -name "*.wav"); do
    run_with_lock resample_delete "$f"
done
