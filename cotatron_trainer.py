import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from omegaconf import OmegaConf

from utils.utils import get_commit_hash
from cotatron import Cotatron
from utils.loggers import TacotronLogger


def main(args):
    model = Cotatron(args)

    hp_global = OmegaConf.load(args.config[0])
    hp_cota = OmegaConf.load(args.config[1])

    hp = OmegaConf.merge(hp_global, hp_cota)

    save_path = os.path.join(hp.log.chkpt_dir, args.name)
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(hp.log.chkpt_dir, args.name),
        monitor='val_loss',
        verbose=True,
        save_top_k=args.save_top_k, # save all
        prefix=get_commit_hash(),
    )

    tb_logger = TacotronLogger(
        save_dir=hp.log.log_dir,
        name=args.name,
    )

    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        default_root_dir=save_path,
        gpus=-1 if args.gpus is None else args.gpus,
        accelerator=None,
        num_sanity_val_steps=1,
        resume_from_checkpoint=args.checkpoint_path,
        gradient_clip_val=hp.train.grad_clip,
        fast_dev_run=args.fast_dev_run,
        check_val_every_n_epoch=args.val_epoch,
        progress_bar_refresh_rate=1,
        max_epochs=10000,
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, nargs=2, required=True,
                        help="path of configuration yaml file")
    parser.add_argument('-g', '--gpus', type=str, default=None,
        help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="Name of the run.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint for resuming")
    parser.add_argument('-s', '--save_top_k', type=int, default=-1,
                        help="save top k checkpoints, default(-1): save all")
    parser.add_argument('-f', '--fast_dev_run', type=bool, default=False,
                        help="fast run for debugging purpose")
    parser.add_argument('--val_epoch', type=int, default=1,
                        help="run val loop every * training epochs")
    args = parser.parse_args()

    main(args)
