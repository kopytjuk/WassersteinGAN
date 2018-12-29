# todo: select via GUI
$config_path = "weights/20181228_2356/generator_config.json"
$weights_path = "weights/20181228_2356/netG_epoch_1914.pth"
$nimages = "5"
python generate.py -c "$config_path" -w "$weights_path" -o generated-images -n "$nimages" --cuda