python train_bayesian.py \
       --workers 4 \
       --lr 0.01 \
       --epochs 50 \
       --gpu-ids 0 \
       --batch-size 32 \
       --test-batch-size 1 \
       --checkname battn_unet \
       --eval-interval 5 \
       --dataset lidc \
       --loss-type ELBO \
       --nchannels 1 \
       --model batten-unet \
       --nclass 1 \
       --task-num 0 \
       --save-path /data/ssd/qingqiao/BOEMD_run_test