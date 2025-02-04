
import argparse
import logging
import sys

from check import CheckCmd
from convert import ConvertCmd
from draw import DrawCmd
from extract import ExtractCmd
from infer_shape import InferShapeCmd
from inspect import InspectCmd
from optimize import OptimizeCmd
from setup import SetupCmd

def dispatch():
    dispatch_core(sys.argv[1:])


def dispatch_core(*raw_args):
    print("Running {}".format(*raw_args))

    parser = argparse.ArgumentParser(description="Onnxcli")
    subparsers = parser.add_subparsers(title='subcommands')

    CheckCmd(subparsers)
    ConvertCmd(subparsers)
    DrawCmd(subparsers)
    ExtractCmd(subparsers)
    InferShapeCmd(subparsers)
    InspectCmd(subparsers)
    OptimizeCmd(subparsers)
    SetupCmd(subparsers)

    args = parser.parse_args(*raw_args)
    args.func(args)

if __name__ == '__main__':
    dispatch()