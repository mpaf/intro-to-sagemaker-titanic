print("We are using this code!")

import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed

import xgboost as xgb


def _xgb_train(params, dtrain, evals, num_boost_round, model_dir):
    """Run xgb train on arguments

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training, or is running single node training job. Note that rabit_run will include this argument.
    """
    booster = xgb.train(params=params, dtrain=dtrain, evals=evals, num_boost_round=num_boost_round)

    model_location = model_dir + '/xgboost-model'
    pkl.dump(booster, open(model_location, 'wb'))
    logging.info("Stored trained model at {}".format(model_location))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--verbose', type=int)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--num_class', type=int)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()

    dtrain = get_dmatrix(args.train, 'csv')
    dval =  get_dmatrix(args.validation, 'csv')
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'num_class': args.num_class,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbose': args.verbose,
        'objective': args.objective}
    
    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir)

    if dtrain:
        _xgb_train(**xgb_train_args)
    else:
        raise ValueError("Training channel must have data to train model.")


def model_fn(model_dir):
    """Deserialized and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = 'xgboost-model'
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster