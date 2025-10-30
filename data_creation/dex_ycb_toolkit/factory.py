# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Factory method for easily getting datasets by name."""

from .dex_ycb import DexYCBDataset
from .dex_ycb_mv import DexYCBDatasetMV

_sets = {}

for setup in ('s0', 's1', 's2', 's3'):
  for split in ('train', 'val', 'test'):
    name = '{}_{}'.format(setup, split)
    _sets[name] = (lambda setup=setup, split=split: DexYCBDataset(setup, split))


# def get_dataset(name):
#   """Gets a dataset by name.

#   Args:
#     name: Dataset name. E.g., 's0_test'.

#   Returns:
#     A dataset.

#   Raises:
#     KeyError: If name is not supported.
#   """
#   if name not in _sets:
#     raise KeyError('Unknown dataset name: {}'.format(name))
#   return _sets[name]()

def get_dataset(name):
  """Gets a dataset by name.

  Args:
    name: Dataset name. E.g., 's0_test'.

  Returns:
    A dataset.

  Raises:
    KeyError: If name is not supported.
  """
  for _name in _sets:
    if _name in name: # input contains the key
      return _sets[_name]()

  raise KeyError('Unknown dataset name: {}'.format(name))


_mv_sets = {}

for setup in ('s0', 's1', 's2', 's3'):
  for split in ('train', 'val', 'test'):
    name = '{}_{}_mv'.format(setup, split)
    _mv_sets[name] = (lambda setup=setup, split=split: DexYCBDatasetMV(setup, split))


def get_dataset_mv(name):
  """Gets a dataset by name.

  Args:
    name: Dataset name. E.g., 's0_test'.

  Returns:
    A dataset.

  Raises:
    KeyError: If name is not supported.
  """
  for _mv_name in _mv_sets:
    if _mv_name in name: # input contains the key
      return _mv_sets[_mv_name]()

  raise KeyError('Unknown dataset name: {}'.format(name))
