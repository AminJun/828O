source ~/virtualenvironment/amin/bin/activate
run_date=`date '+%d-%H:%M'`
log_file="./logs/${run_date}"
mkdir -p ${log_file}
echo "Run 1"
CUDA_VISIBLE_DEVICES=$1 python train.py 2 true c 1> ${log_file}/output.txt 2> ${log_file}/log.txt
echo "Run 2"
CUDA_VISIBLE_DEVICES=$1 python train.py 2 true h 1> ${log_file}/output.txt 2> ${log_file}/log.txt
echo "Run 3"
CUDA_VISIBLE_DEVICES=$1 python train.py 2 true b 1> ${log_file}/output.txt 2> ${log_file}/log.txt
echo "Run 4"
CUDA_VISIBLE_DEVICES=$1 python train.py 2 false c 1> ${log_file}/output.txt 2> ${log_file}/log.txt
echo "Run 5"
CUDA_VISIBLE_DEVICES=$1 python train.py 2 false h 1> ${log_file}/output.txt 2> ${log_file}/log.txt
echo "Run 6"
CUDA_VISIBLE_DEVICES=$1 python train.py 2 false b 1> ${log_file}/output.txt 2> ${log_file}/log.txt
deactivate
