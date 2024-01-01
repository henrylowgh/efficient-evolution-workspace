import os
from amis import reconstruct_multi_models

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Recommend substitutions to a wildtype sequence')
    parser.add_argument('sequence', type=str, help='Wildtype sequence')
    parser.add_argument('--model-names', type=str, default=['esm1b', 'esm1v1', 'esm1v2', 'esm1v3', 'esm1v4', 'esm1v5'], nargs='+', help='Type of language model (e.g., esm1b, esm1v1)')
    parser.add_argument('--alpha', type=float, default=None, help='alpha stringency parameter')
    parser.add_argument('--cuda', type=str, default='cuda', help='cuda device to use')
    args = parser.parse_args()
    return args

# Updated shorten_model_name function
def shorten_model_name(full_name):
    if 'esm1b' in full_name:
        return '1b'
    elif 'esm1v' in full_name:
        version = full_name.rsplit('_', 1)[-1]
        return f'v{version}'
    return full_name

if __name__ == '__main__':
    args = parse_args()

    if ":" in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda.split(':')[-1]

    mutations_models, mutations_model_names = reconstruct_multi_models(
        args.sequence,
        args.model_names,
        alpha=args.alpha,
        return_names=True
    )

    for mutation, count in sorted(mutations_models.items(), key=lambda item: -item[1]):
        mutation_str = f'{mutation[1]}{mutation[0] + 1}{mutation[2]}'
        model_names = ', '.join([shorten_model_name(name) for name in mutations_model_names[mutation]])
        print(f'{mutation_str}\t{count} | ({model_names})')
