TRAIN_FILE=train.py
while getopts ":d:g:c:" opt; do
  case ${opt} in
  d)
    CUDA_VISIBLE_DEVICES=$OPTARG
    ;;
  g)
    nproc_per_node=$OPTARG
    ;;
  c)
    config=$OPTARG
    ;;
  \?)
    echo "Invalid Option: -$OPTARG" 1>&2
    exit 1
    ;;
  esac
done


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m torch.distributed.launch --master_port 29502 --nproc_per_node=$nproc_per_node $TRAIN_FILE --config $config --launcher pytorch


# sh dist_train.sh -d 2,3 -g 2  -c ../config/train_config.py