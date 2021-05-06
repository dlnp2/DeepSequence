import argparse
import os

import pandas as pd


def update_variant(seq, wt_seq, variant, deleted_cols):
    # We assume single mutants
    wt_aa = variant[0]
    mt_aa = variant[-1]
    _seq = "".join([aa for _pos, aa in enumerate(list(seq)) if _pos not in deleted_cols])
    _wt_seq = "".join([aa for _pos, aa in enumerate(list(wt_seq)) if _pos not in deleted_cols])
    mutated_pos = -1
    for pos, (_mt_aa, _wt_aa) in enumerate(zip(_seq, _wt_seq)):
        if _mt_aa != _wt_aa:
            mutated_pos = pos
    if mutated_pos == -1:
        return None, None
    else:
        return wt_aa + str(mutated_pos + 1) + mt_aa, _seq


def update_records(protein_seq, deleted_cols, verbose=False):
    """Update variant descriptions, deleting mutants on the specified columns"""
    wt_seq = protein_seq.loc[protein_seq["mut_type"] == "synonymous", "sequence"].values[0]

    records = []
    for record in protein_seq.itertuples():
        variant = record.Variant
        new_variant, new_seq = update_variant(record.sequence, wt_seq, variant, deleted_cols)
        if new_variant is None:
            if verbose:
                print(
                    "Deleting variant {} because the mutation is on a deleted column."
                    .format(variant)
                    )
        else:
            records.append([new_variant, record.scaled_effect1, new_seq])

    return pd.DataFrame(records, columns=["Variant", "scaled_effect1", "sequence"])


def main(args):
    print("protein_seq_csv: {}".format(args.protein_seq_csv))
    print("deleted_cols_file: {}".format(args.deleted_cols_file))
    print("output_dir: {}".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.deleted_cols_file) as fin:
        deleted_cols = [int(line.strip()) for line in fin]

    protein_seq = pd.read_csv(args.protein_seq_csv)
    org_len = len(protein_seq)
    records = update_records(protein_seq, deleted_cols, verbose=args.verbose)
    new_len = len(records)
    print(
        "Left with {:,} sequences (of {:,}, {:.3} %)."
        .format(new_len, org_len, 100 * float(new_len) / org_len)
    )
    records.to_csv(
        os.path.join(args.output_dir, os.path.basename(args.protein_seq_csv)),
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("protein_seq_csv", type=str)
    parser.add_argument("deleted_cols_file", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())