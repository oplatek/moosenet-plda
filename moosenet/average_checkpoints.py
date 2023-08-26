#!/usr/bin/env python3
# Copied and modified from fairseq scripts
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging

import torch


def load_avg_model(ckpt_paths, prefer_ckpt=True, **model_kwargs):
    from moosenet.models import MOSModelMixin

    model_dict = average_checkpoints_paths(ckpt_paths, model_key="state_dict")
    load_model_strictly = lambda strict: MOSModelMixin.load_from_unpickled_checkpoint(
        model_dict, strict=strict, prefer_ckpt=prefer_ckpt, **model_kwargs
    )
    try:
        model = load_model_strictly(True)
        logging.info("The model loaded with strict loading")
    except Exception as e:
        logging.debug(f"Loading the model strictly FAILED: INSPECT CAREFULLY!:{str(e)}")
        model = load_model_strictly(False)
        logging.warning("Without strict loading the model loaded successfully")
    return model


def average_checkpoints_paths(inputs, model_key):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """

    state_dicts = []
    new_state = None
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        if new_state is None:
            new_state = state
        state_dicts.append(state[model_key])

    model = average_checkpoints(state_dicts)
    new_state[model_key] = model
    return new_state


def average_checkpoints(state_dicts):
    """Avarge state_dicts, Return state_dict  with averaged weights.

    Args:
      inputs: Checkpoint "state_dict"; Mapping of [str, tensor]

    Returns:
      Returned dict corresponds to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    num_models = len(state_dicts)

    for i, state_dict in enumerate(state_dicts):
        model_params = state_dict

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"For {i}. checkpoint , expected list of params: {params_keys}, "
                f"but found: {model_params_keys}"
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    return averaged_params
