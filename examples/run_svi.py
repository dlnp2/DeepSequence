import numpy as np
import time
import sys
sys.path.insert(0, "../DeepSequence/")
import model
import helper
import train
import argparse
import os

data_params = {
    "dataset"           :   "BLAT_ECOLX"
    }

model_params = {
    "bs"                :   100,
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

train_params = {
    "num_updates"       :   300000,
    "save_progress"     :   True,
    "verbose"           :   True,
    # "save_parameters"   :   False,
    "save_parameters"   :   30000,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment_file", type=str, default="")
    parser.add_argument("--working_dir", type=str, default=".")
    args = parser.parse_args()

    if args.alignment_file != "":
        data_params["dataset"] = os.path.basename(args.alignment_file).split(".")[0]

    data_helper = helper.DataHelper(
        dataset=data_params["dataset"],
        calc_weights=True,
        alignment_file=args.alignment_file,
        working_dir=args.working_dir
    )

    n_seqs = data_helper.x_train.shape[0]
    if n_seqs < model_params["bs"]:
        msg = "Found {} sequences less than batch size: {}.".format(n_seqs, model_params["bs"])
        msg += " Setting to {}.".format(n_seqs)
        print(msg)
        model_params["bs"] = n_seqs

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

    job_string = helper.gen_job_string(data_params, model_params)

    print (job_string)

    train.train(data_helper, vae_model,
        num_updates             =   train_params["num_updates"],
        save_progress           =   train_params["save_progress"],
        save_parameters         =   train_params["save_parameters"],
        verbose                 =   train_params["verbose"],
        job_string              =   job_string)

    vae_model.save_parameters(file_prefix=job_string)
