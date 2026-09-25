"""Show resolved output locations without creating directories."""

import argparse
import json
from dataclasses import asdict

from hepml.adapters.configuration import output_paths


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps({key: str(path.resolve()) for key, path in asdict(output_paths()).items()}, indent=2))
