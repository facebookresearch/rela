sbatch --job-name GAMEasterix_SAME_HID0_LR0.0001_EPS1e-05 \
       --mem 500G \
       --partition learnfair \
       --constraint pascal \
       --gres gpu:2 \
       --nodes 1 \
       --ntasks-per-node 1 \
       --cpus-per-task 80 \
       --time 2880 \
       --output /private/home/hengyuan/rela/pyrela/sweep2/r2d2_newnet/GAMEasterix_SAME_HID0_LR0.0001_EPS1e-05/stdout.log \
       --error /private/home/hengyuan/rela/pyrela/sweep2/r2d2_newnet/GAMEasterix_SAME_HID0_LR0.0001_EPS1e-05/stderr.log \
       /private/home/hengyuan/rela/pyrela/sweep2/r2d2_newnet/GAMEasterix_SAME_HID0_LR0.0001_EPS1e-05/train.sh
