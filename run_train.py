from engine import Trainer


if __name__ == '__main__':
    trainer = Trainer(batch_size=96)
    trainer.train()