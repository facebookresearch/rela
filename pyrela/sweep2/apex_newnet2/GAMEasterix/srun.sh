sbatch --job-name GAMEasterix \
       --mem 500G \
       --partition learnfair \
       --constraint pascal \
       --gres gpu:2 \
       --nodes 1 \
       --ntasks-per-node 1 \
       --cpus-per-task 80 \
       --time 2880 \
       --output /private/home/hengyuan/rela/pyrela/sweep2/apex_newnet2/GAMEasterix/stdout.log \
       --error /private/home/hengyuan/rela/pyrela/sweep2/apex_newnet2/GAMEasterix/stderr.log \
       /private/home/hengyuan/rela/pyrela/sweep2/apex_newnet2/GAMEasterix/train.sh
