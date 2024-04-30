


if __name__ == '__main__':
    import os
    import argparse

    from basicsr.utils import scandir

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str)
    parser.add_argument('-suffix', '--suffix', type=str, default=None)
    args = parser.parse_args()

    all_targets = list(scandir(args.input_dir, recursive=True, suffix=args.suffix))
    print(all_targets, len(all_targets))
    all_targets = [i for i in all_targets if '.ipynb_checkpoints' not in i]

    if args.output_file is not None:
        print(f'Wrting to {args.output_file}')
        with open(args.output_file, 'w') as f:
            f.write('\n'.join(all_targets))
