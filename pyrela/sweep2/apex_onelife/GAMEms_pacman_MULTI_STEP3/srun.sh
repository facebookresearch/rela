sbatch --job-name GAMEms_pacman_MULTI_STEP3 \
       --mem 500G \
       --partition learnfair \
       --constraint pascal \
       --gres gpu:2 \
       --nodes 1 \
       --ntasks-per-node 1 \
       --cpus-per-task 80 \
       --time 2880 \
       --output /private/home/hengyuan/rela/pyrela/sweep2/apex_onelife/GAMEms_pacman_MULTI_STEP3/stdout.log \
       --error /private/home/hengyuan/rela/pyrela/sweep2/apex_onelife/GAMEms_pacman_MULTI_STEP3/stderr.log \
       /private/home/hengyuan/rela/pyrela/sweep2/apex_onelife/GAMEms_pacman_MULTI_STEP3/train.sh
