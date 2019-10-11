sbatch --job-name GAMEbowling_SAME_HID0 \
       --mem 500G \
       --partition learnfair \
       --constraint pascal \
       --gres gpu:2 \
       --nodes 1 \
       --ntasks-per-node 1 \
       --cpus-per-task 80 \
       --time 2880 \
       --output /private/home/hengyuan/rela/pyrela/sweep2/r2d2_onelife_net/GAMEbowling_SAME_HID0/stdout.log \
       --error /private/home/hengyuan/rela/pyrela/sweep2/r2d2_onelife_net/GAMEbowling_SAME_HID0/stderr.log \
       /private/home/hengyuan/rela/pyrela/sweep2/r2d2_onelife_net/GAMEbowling_SAME_HID0/train.sh
