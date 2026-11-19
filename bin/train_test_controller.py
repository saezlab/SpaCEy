import argparse
import os
import sys
from types import SimpleNamespace

import custom_tools as custom_tools
from trainer_tester import trainer_tester

extra_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # "--en" must not be abbreviated
extra_parser.add_argument("--cv_setting", choices=["nfold", "heldout"], default="nfold",
                       help="nfold (default) = the paper's CV, the held-out fold is also used for early "
                            "stopping and checkpoint choice (optimistic); heldout = early stopping on an inner split of "
                            "the training fold + refit, held-out fold scored once")
extra_parser.add_argument("--folds_json", default=None, help="split file, fold_k = [train image ids, test image ids]")
extra_parser.add_argument("--params_json", default=None,
                       help="hyperparameter file (key 'hyperparameters'); overrides the CLI hyperparameters")
extra_parser.add_argument("--min_epochs", type=int, default=0, help="smallest number of epochs to choose")
extra_parser.add_argument("--augment", action="store_true", help="graph augmentation during training")
extra_parser.add_argument("--feature_normalise", choices=["raw", "log1p_z"], default="raw",
                       help="raw (default) = protein abundances as stored, no normalisation; "
                            "log1p_z = log1p + z-score with training-fold statistics")
extra_parser.add_argument("--seed", type=int, default=42, help="fold k is trained with seed SEED + k")
extra_args, rest = extra_parser.parse_known_args()
sys.argv = [sys.argv[0]] + rest  # general_parser parses sys.argv and rejects unknown options

custom_tools.set_seeds(42, deterministic=False)
parser_args = custom_tools.general_parser()
for key, value in vars(extra_args).items():
    setattr(parser_args, key, value)
setup_args = SimpleNamespace()

setup_args.id = custom_tools.generate_session_id()

print(f"Session id: {setup_args.id}")

setup_args.S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
setup_args.RESULT_PATH = os.path.join(setup_args.S_PATH, "../results", "idedFiles", parser_args.en)
setup_args.MODEL_PATH = os.path.join(setup_args.S_PATH, "../models", parser_args.en)

custom_tools.create_directories([setup_args.RESULT_PATH, setup_args.MODEL_PATH])

trainer_tester(parser_args, setup_args)
