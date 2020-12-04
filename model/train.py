import pytorch_lightning as pl
import argparse

from model import GuitarModel
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    model = GuitarModel(args)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, checkpoint_callback=ModelCheckpoint(
            filepath='model/guitar_model.ckpt',
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels", type=int, default=16)
    parser.add_argument("--dilation_depth", type=int, default=8)
    parser.add_argument("--num_repeat", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--max_epochs", type=int, default=1_500)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--data", default="../data/training/data.pickle")
    args = parser.parse_args()
    main(args)
