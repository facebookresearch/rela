sbatch --job-name GAMEbowling_SAME_HID0_LR1.25e-06 \
       --mem 500G \
       --partition learnfair \
       --constraint pascal \
       --gres gpu:2 \
       --nodes 1 \
       --ntasks-per-node 1 \
       --cpus-per-task 80 \
       --time 2880 \
       --output /private/home/hengyuan/rela/pyrela/sweep2/r2d2_onelife/GAMEbowling_SAME_HID0_LR1.25e-06/stdout.log \
       --error /private/home/hengyuan/rela/pyrela/sweep2/r2d2_onelife/GAMEbowling_SAME_HID0_LR1.25e-06/stderr.log \
       /private/home/hengyuan/rela/pyrela/sweep2/r2d2_onelife/GAMEbowling_SAME_HID0_LR1.25e-06/train.sh
