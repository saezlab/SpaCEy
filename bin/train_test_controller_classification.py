
import torch
from data_processing import OUT_DATA_PATH
import os
import custom_tools
from types import SimpleNamespace
from loss import CoxPHLoss, NegativeLogLikelihood
from trainer_tester_classification import trainer_tester


custom_tools.set_seeds(42, deterministic=False)
parser_args = custom_tools.general_parser()
setup_args = SimpleNamespace()

setup_args.id = custom_tools.generate_session_id()

print(f"Session id: {setup_args.id}")

setup_args.S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
# setup_args.OUT_DATA_PATH = os.path.join(setup_args.S_PATH, "../data", "out_data", parser_args.en)
setup_args.RESULT_PATH = os.path.join(setup_args.S_PATH, "../results", "idedFiles", parser_args.en)
setup_args.PLOT_PATH = os.path.join(setup_args.S_PATH, "../plots", parser_args.en)
setup_args.MODEL_PATH = os.path.join(setup_args.S_PATH, "../models", parser_args.en)

custom_tools.create_directories([setup_args.RESULT_PATH, setup_args.PLOT_PATH, setup_args.MODEL_PATH])

# setup_args.T2VT_ratio = 4
# setup_args.V2T_ratio = 1
setup_args.use_fold = parser_args.fold


# This is NOT for sure, loss can change inside the class
setup_args.criterion = None
if parser_args.loss=="MSE":
    setup_args.criterion = torch.nn.MSELoss()
elif parser_args.loss=="Huber":
    setup_args.criterion = torch.nn.HuberLoss()
elif parser_args.loss == "CoxPHLoss":
    setup_args.criterion = CoxPHLoss()
elif parser_args.loss== "NegativeLogLikelihood":
    print("Loss is NegativeLogLikelihood")
    # setup_args.criterion = NegativeLogLikelihood()
    setup_args.criterion = torch.nn.BCEWithLogitsLoss()
elif parser_args.loss== "BCEWithLogitsLoss":
    print("Loss is BCEWithLogitsLoss")
    setup_args.criterion = torch.nn.BCEWithLogitsLoss()
else:
    raise Exception("Loss function should be MSE, Huber, CoxPHLoss, NegativeLogLikelihood, or BCEWithLogitsLoss...")

setup_args.print_every_epoch = 10
setup_args.plot_result = True

# print(parser_args)
# print(setup_args)
# Object can be saved if wanted
trainer_tester(parser_args, setup_args)





# python train_test_controller_classification.py --dataset_name Lung --bs 16 --dropout 0.3 --en GATV2_CoxPHLoss_Lung --epoch 1000 --factor 0.5 --fcl 128 --gcn_h 32 --gpu_id 0 --heads 1 --loss NegativeLogLikelihood --lr 0.1 --min_lr 0.0001 --model GATV2 --no-fold  --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --unit month --weight_decay 1e-05
# python train_test_controller_classification.py --dataset_name Lung --bs 32 --dropout 0.3 --en GATV2_CoxPHLoss_month_Lung --epoch 1000 --factor 0.2 --fcl 256 --gcn_h 32 --gpu_id 0 --heads 5 --loss NegativeLogLikelihood --lr 0.0001 --min_lr 0.0001 --model GATV2 --no-fold  --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 10 --unit month --weight_decay 3e-06