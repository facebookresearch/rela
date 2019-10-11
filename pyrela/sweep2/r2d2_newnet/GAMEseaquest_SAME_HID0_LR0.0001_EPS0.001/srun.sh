sbatch --job-name GAMEseaquest_SAME_HID0_LR0.0001_EPS0.001 \
       --mem 500G \
       --partition learnfair \
       --constraint pascal \
       --gres gpu:2 \
       --nodes 1 \
       --ntasks-per-node 1 \
       --cpus-per-task 80 \
       --time 2880 \
       --output /private/home/hengyuan/rela/pyrela/sweep2/r2d2_newnet/GAMEseaquest_SAME_HID0_LR0.0001_EPS0.001/stdout.log \
       --error /private/home/hengyuan/rela/pyrela/sweep2/r2d2_newnet/GAMEseaquest_SAME_HID0_LR0.0001_EPS0.001/stderr.log \
       /private/home/hengyuan/rela/pyrela/sweep2/r2d2_newnet/GAMEseaquest_SAME_HID0_LR0.0001_EPS0.001/train.sh
