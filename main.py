from __future__ import absolute_import, division, print_function

from options import Options

options = Options()
opts = options.parse()
from trainer import *

if __name__ == '__main__':
    trainer = Trainer(opts)

    if opts.test_only:
        if opts.resume is None:
            raise RuntimeError("=> no checkpoint found...")
        else:
            print("checkpoint found at '{}'" .format(opts.resume))
        trainer.test()
    else:
        for epoch in range(trainer.opts.start_epoch, trainer.opts.epochs):
            trainer.train()
            trainer.validate()
            trainer.scheduler.step()
            trainer.cur_epochs += 1

        print('=> End training\n\n')
        trainer.writer.close()
