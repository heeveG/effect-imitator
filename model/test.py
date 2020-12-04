import pickle
import torch
from scipy.io import wavfile
import argparse
import numpy as np

from model import GuitarModel


@torch.no_grad()
def test(args):
    model = GuitarModel.load_from_checkpoint(args.model)
    model.eval()
    data = pickle.load(open(args.data, "rb"))

    x_test = np.concatenate(
        (np.concatenate((np.zeros_like(data["x_test"][0:1]), data["x_test"][:-1]), axis=0), data["x_test"]),
        axis=2)

    x_split = np.array_split(x_test, 10)
    y_pred = np.concatenate([model(torch.from_numpy(i)).numpy() for i in x_split])

    y_pred = y_pred[:, :, -data["x_test"].shape[2]:] * 10000

    wavfile.write("predicted.wav", 44100, y_pred.flatten().astype(np.int16))

    raw = data["x_test"] * data["std"] + data["mean"]
    wavfile.write("raw.wav", 44100, raw.flatten().astype(np.int16))

    effect = data["y_test"]
    wavfile.write("original_effect.wav", 44100, effect.flatten().astype(np.int16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/guitarmodel.ckpt")
    parser.add_argument("--data", default="../data/training/data.pickle")
    args = parser.parse_args()
    test(args)
