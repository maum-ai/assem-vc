# Hard-coded for convert metadata into phoneme

import tqdm
from g2p_en import G2p
from argparse import ArgumentParser


def write_metadata(metadata, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        for m in metadata:
            if m is None:
                continue
            f.write('|'.join([str(x) for x in m]) + '\n')


def load_metadata(path, split="|"):
    with open(path, 'r', encoding='utf-8') as f:
        metadata = [line.strip().split(split) for line in f]

    return metadata

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_filename', type=str, required=True,
                        help="name of the input file")
    parser.add_argument('-o', '--output_filename', type=str, required=True,
                        help="name of the output file")
    args = parser.parse_args()

    meta = load_metadata(args.input_filename)
    g2p = G2p()

    for idx, (audiopath, text, spk_id) in enumerate(tqdm.tqdm(meta)):
        phoneme = g2p(text)
        converted = ['{']
        for x in phoneme:
            if x==' ':
                converted.append('}')
                converted.append('{')
            elif x=='-':
                continue
            else:
                converted.append(x)

        converted.append('}')
        phoneme = " ".join(str(x) for x in converted)
        phoneme = phoneme.replace(' }', '}').replace('{ ','{')
        phoneme = phoneme.replace('0','').replace('1','').replace('2','').replace('{\'}','\'').replace('{...}','...')
        meta[idx][1] = phoneme.replace(' {!}','!').replace(' {?}','?').replace(' {.}','.').replace(' {,}',',')

    write_metadata(meta, args.output_filename)

