"""
Copyright 2017 The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = 'jhuapl'
__version__ = 0.1
# Modified by: Rodrigo Minetto (UTFPR-Brazil Assistant Professor/USF-Tampa-USA Post-doctoral)

from fmowBaseline import FMOWBaseline
import params
import argparse as ap 

def main (parser):
    baseline = FMOWBaseline(params, parser)
    if (parser['train'] and not parser['prepare']):
        baseline.train_cnn()
    if (parser['test'] and not parser['prepare']):
        baseline.test_models()
    
if __name__ == "__main__":
    arg = ap.ArgumentParser()
    arg.add_argument ("--algorithm", default="", type=str)
    arg.add_argument ("--train", default="", type=str)
    arg.add_argument ("--test", default="", type=str)
    arg.add_argument ("--nm", default="", type=str)
    arg.add_argument ("--prepare", default="", type=str)
    arg.add_argument ("--num_gpus", default=1, type=int)
    arg.add_argument ("--num_epochs", default=12, type=int)
    arg.add_argument ("--batch_size", default=64, type=int)
    arg.add_argument ("--load_weights", default="", type=str)
    arg.add_argument ("--fine_tunning", default="", type=str)
    arg.add_argument ("--class_weights", default="", type=str)
    arg.add_argument ("--prefix", default="", type=str)
    arg.add_argument ("--generator", default="", type=str)
    arg.add_argument ("--database", default="", type=str)
    arg.add_argument ("--path", default="", type=str)
    parser = vars(arg.parse_args())
    print (parser)
    main (parser)
