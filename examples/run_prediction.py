import argparse
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "../DeepSequence")
import model, helper


data_params = {}
model_params = {
    "bs"                :   50,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   12345,
    "conv_pat"          :   True,
    "d_c_size"          :   40
    }


def main(args):
    if args.alignment_file != "":
        data_params["dataset"] = os.path.basename(args.alignment_file).split(".")[0]

    data_helper = helper.DataHelper(
        dataset=data_params["dataset"],
        calc_weights=False,
        alignment_file=args.alignment_file,
        working_dir=args.working_dir,
        load_all_sequences=False
    )

    vae_model   = model.VariationalAutoencoder(data_helper,
        batch_size                     =   model_params["bs"],
        encoder_architecture           =   [model_params["encode_dim_zero"],
                                                model_params["encode_dim_one"]],
        decoder_architecture           =   [model_params["decode_dim_zero"],
                                                model_params["decode_dim_one"]],
        n_latent                       =   model_params["n_latent"],
        logit_p                        =   model_params["logit_p"],
        sparsity                       =   model_params["sparsity"],
        encode_nonlinearity_type       =   "relu",
        decode_nonlinearity_type       =   "relu",
        final_decode_nonlinearity      =   model_params["final_decode_nonlin"],
        final_pwm_scale                =   model_params["final_pwm_scale"],
        conv_decoder_size              =   model_params["d_c_size"],
        convolve_patterns              =   model_params["conv_pat"],
        n_patterns                     =   model_params["n_pat"],
        random_seed                    =   model_params["r_seed"],
        working_dir                    =   args.working_dir
        )

    print("Loading model parameters...")
    vae_model.load_parameters(file_prefix=args.ckpt_path)
    print("Computing delta elbo...")
    mutant_name_list, delta_elbo_list = data_helper.single_mutant_matrix(
        vae_model, N_pred_iterations=args.n_iters)
    print("Done.")

    # Result summary
    outprefix = os.path.join(args.working_dir, os.path.basename(args.alignment_file).split(".")[0])

    preds = pd.DataFrame(
        zip(mutant_name_list, delta_elbo_list),
        columns=["Variant", "delta_elbo"]
    )
    preds.to_csv(outprefix + "_preds_raw.csv",index=False)
    preds.loc[:, "Variant"] = preds["Variant"].apply(
        lambda s: s[0] + str(int(s[1:-1]) + args.start_idx - 1) + s[-1]
    )
    variant_seq = pd.read_csv(args.variant_seq_csv)
    variant_seq = variant_seq.loc[variant_seq["Variant"].apply(lambda s: s[0] != s[-1])]
    preds = preds.loc[preds["Variant"].isin(variant_seq["Variant"])]
    org_len = len(variant_seq)
    variant_seq = variant_seq.merge(preds, on="Variant", how="left")
    assert len(variant_seq) == org_len
    variant_seq[["Variant", "scaled_effect1", "delta_elbo"]].to_csv(outprefix + ".csv", index=False)
    print("Saved results to {}".format(outprefix + ".csv"))

    correlation, _ = spearmanr(
        variant_seq["scaled_effect1"].values,
        variant_seq["delta_elbo"].values
    )
    print("Spearman's correlation: {}".format(correlation))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("alignment_file", type=str)
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("variant_seq_csv", type=str)
    parser.add_argument("start_idx", type=int)
    parser.add_argument("--working_dir", type=str, default=".")
    parser.add_argument("--n_iters", type=int, default=1000)
    args = parser.parse_args()
    main(args)