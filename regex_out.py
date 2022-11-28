import re

file_path_list = [
    'run_reddit_seed.sh.551788.out',
    'run_reddit_seed_gdc.sh.551790.out',
    'run_yelp_seed.sh.551769.out',
    'run_yelp_seed_gdc.sh.551787.out',
    ]
cols = ['dataset',
            'agp',
            'α',
            't',
            'test acc',
            'seed',
            'L',
            'rmax(εの逆数)',
            'train propagation time',
            'train clock time',
            'full clock time',
            'full propagation time',
            'lr',
            'dropout',
            'hidden',
            'batch',
            'layer',
            'Train cost',
            'log file']


def get_data_list(file_path):

    with open(file_path, mode='r') as f:
        text=f.read()

    text_list = text.split('--------------------------\nos.cpu_count:')


    regex = re.compile(
        r"Namespace\(L=(\d+), agp_alg='(\w+)', alpha=(\d+\.?\d*), batch=(\d+), bias='bn', dataset='(\w+)', dev=None, dropout=(\d+\.?\d*), epochs=(\d+), hidden=(\d+), layer=(\d+), lr=(\d+\.?\d*), patience=(\d+), rmax=(.+), seed=(\d+), ti=(\d+\.?\d*), weight_decay=(\d+\.?\d*)\)"
        )
    regex2 = re.compile(r"The propagation time: (\d+\.?\d*) s\s?\nThe clock time : (\d+\.?\d*) s\s?\nFor full features propagation: culculating\.\.\.\s?\nBegin propagation\.\.\.\s?\nThe propagation time: (\d+\.?\d*) s\s?\nThe clock time : (\d+\.?\d*) s")
    regex3 = re.compile(r"Train cost: (\d+\.?\d*)s\s?\nLoad (\d+)th epoch\s?\nTest accuracy:(\d+\.?\d*)%")

    ret = []

    for i, text in enumerate(text_list):
        regex_result = regex.search(text)
        regex_result2 = regex2.search(text)
        regex_result3 = regex3.search(text)

        result_dict = {
            'dataset': regex_result.group(5),
            'agp': regex_result.group(2),
            'α': regex_result.group(3) if regex_result.group(2)=='appnp_agp' else "4",
            't': regex_result.group(14) if regex_result.group(2)=='gdc_agp' else f"0.9",
            'test acc':regex_result3.group(3),
            'seed': regex_result.group(13),
            "L": regex_result.group(1),
            'rmax(εの逆数)': regex_result.group(12),
            'train propagation time': regex_result2.group(1),
            'train clock time': regex_result2.group(2),
            'full clock time': regex_result2.group(3),
            'full propagation time': regex_result2.group(4),
            'lr': regex_result.group(10),
            'dropout': regex_result.group(6),
            'hidden': regex_result.group(8),
            'batch': regex_result.group(4),
            'layer': regex_result.group(9),
            'Train cost': regex_result3.group(1),
            'log file': file_path,
        }
        ret.append(result_dict)
    return ret


result = []
for file_path in file_path_list:
    result.extend(get_data_list(file_path))

with open('regex_out.csv', mode='w') as f:
    f.write(",".join(cols) + '\n')
    for result_dict in result:
        for col in cols:
            f.write(result_dict[col].strip() + ', ')
        f.write('\n')
