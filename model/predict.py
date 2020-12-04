import pickle
import torch
from tqdm import tqdm
from scipy.io import wavfile
import argparse
import numpy as np

from model.models.guitar_model import GuitarModel


@torch.no_grad()
def predict(args):
    model = GuitarModel.load_from_checkpoint(args.model)
    model.eval()
    train_data = pickle.load(open(args.train_data, "rb"))

    mean, std = train_data["mean"], train_data["std"]

    in_rate, in_data = wavfile.read(args.input)
    sample_size = int(in_rate * args.sample_time)
    length = len(in_data) - len(in_data) % sample_size

    in_data = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    in_data = (in_data - mean) / std

    prev_sample = np.concatenate((np.zeros_like(in_data[0:1]), in_data[:-1]), axis=0)
    pad_in_data = np.concatenate((prev_sample, in_data), axis=2)

    pred = []
    for x in tqdm(np.array_split(pad_in_data, 1)):
        y = model(torch.from_numpy(x)).numpy()
        pred.append(y)

    pred = np.concatenate(pred)
    pred = pred[:, :, -in_data.shape[2]:] * 10000
    wavfile.write(args.output, 44100, pred.flatten().astype(np.int16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/guitarmodel.ckpt")
    parser.add_argument("--train_data", default="../data/training/data.pickle")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("--hparams", type=str)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    predict(args)
