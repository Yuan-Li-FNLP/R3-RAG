srun -p a800 --nodes=2 --gres=gpu:2 --ntasks=2 --ntasks-per-node=1 --nodelist=slurmd-6,slurmd-10  --cpus-per-task=8 \
bash -c torchrun --nnodes 2 --nproc_per_node 2 \
--rdzv_id $RANDO-M --rdzv_backend c10d --rdzv_endpoint slurmd-6:29549 \
a.py > a.log
