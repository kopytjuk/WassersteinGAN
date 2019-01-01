$experiments_path = "./experiments/1"
try {
    python main.py --dataset cifar10 --dataroot ./data --experiment "$experiments_path"  --save-image-modulo 5 --batchSize 1 --workers 4 --niter 100 --imageSize 64 --n_extra_layers 4 --debug_extra_layers --cuda

}
catch {
    Remove-Item –path "$experiments_path" –recurse
}