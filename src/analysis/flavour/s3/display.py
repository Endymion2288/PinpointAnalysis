"""Reuse native detector plotting with S3 four-class error groups and scores."""
import argparse
from pathlib import Path

from analysis.flavour.s2.display import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--evaluation-tag', default='evaluation_final')
    parser.add_argument('--pages', type=int, default=20)
    args = parser.parse_args()
    args.groups_file = args.root / args.evaluation_tag / 'error_groups.parquet'
    args.all_class_scores = True
    run(args)


if __name__ == '__main__':
    main()
