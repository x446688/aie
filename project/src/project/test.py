from .datagen import *
import os
from pathlib import Path
def run_all_tests():
    # print()
    display_series(gen_base_series(), Path(os.path.dirname(__file__) + "/../../data"))