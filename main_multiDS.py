import os
import argparse
import random
import copy
import setupGC
from training import *
import torch
from tensorboardX import SummaryWriter
from pathlib import Path
import numpy as np

def process_fedSSP(args, clients, server, summary_writer):
    print("\nDone setting up FedSSP devices.")

    print("Running FedSSP ...")
    frame = run_fedSSP(args, clients, server, args.num_rounds, args.local_epoch, samp=None,
                       summary_writer=summary_writer)

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedSSP_{args.type_init}_GC.csv')
    else:
        outfile = os.path.join(outpath, f'{args.repeat}_accuracy_fedSSP_{args.type_init}_GC.csv')

    file_exists = os.path.isfile(outfile)

    frame.to_csv(outfile, mode='a', header=not file_exists)

    print(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu',
                        help='CPU / GPU device.')
    parser.add_argument('--alg', type=str, default='fedSSP',
                        help='Name of algorithms.')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.005, help='lr for preference module')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=1)
    parser.add_argument('--datapath', type=str, default='./Data1',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='chem', choices=['chem', "biochem", 'biochemsn', "biosncv", "chemsn", "chemsncv", "chemcv"])
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=5)
    parser.add_argument('--n_rw', type=int, default=16,
                        help='Size of position encoding (random walk).')
    parser.add_argument('--n_dg', type=int, default=16,
                        help='Size of position encoding (max degree).')
    parser.add_argument('--n_ones', type=int, default=16,
                        help='Size of position encoding (ones).')
    parser.add_argument('--type_init', help='the type of positional initialization',
                        type=str, default='rw_dg', choices=['rw', 'dg', 'rw_dg', 'ones'])
    parser.add_argument('-mo', "--momentum", type=float, default=0.5)
    parser.add_argument('-tau', "--tau_weight", type=float, default=100.0)
    parser.add_argument('-head', "--head", type=float, default=4)
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    outpath = os.path.join(args.outbase, 'raw', args.data_group)
    Path(outpath).mkdir(parents=True, exist_ok=True)

    print("Preparing data ...")
    splitedData, df_stats = setupGC.prepareData_multiDS(args, args.datapath, args.data_group, args.batch_size, seed=seed_dataSplit)
    print("Done")
    if 'fedSSP' in args.alg:
        init_clients, init_server, init_idx_clients = setupGC.setup_devices_SSP(splitedData, args)
    else:
        init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up devices.")

    if 'fedSSP' in args.alg:
        sw_path = os.path.join(args.outbase, 'raw', 'tensorboard', f'{args.data_group}_{args.alg}_{args.type_init}_{args.repeat}')
    else:
        sw_path = os.path.join(args.outbase, 'raw', 'tensorboard', f'{args.data_group}_{args.alg}_{args.repeat}')
    summary_writer = SummaryWriter(sw_path)

    if args.alg == 'fedSSP':
        process_fedSSP(args, clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), summary_writer=summary_writer)


