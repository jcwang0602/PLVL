from .PLVL import PLVL


def build_model(args):
    if args.model_name == "PLVL":
        return PLVL(args)
