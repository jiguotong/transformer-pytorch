import torch
import numpy as np
from tqdm import tqdm

from os.path import dirname, abspath, join, exists
from os import makedirs
from datetime import datetime
import json

PAD_INDEX = 0

BASE_DIR = dirname(abspath(__file__))


class EpochSeq2SeqTrainer:

    def __init__(self, model,
                 train_dataloader, val_dataloader,
                 loss_function, optimizer,
                 logger, run_name,
                 config):

        self.config = config
        self.device = torch.device(self.config['device'])

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.loss_function = loss_function
        # self.metric_function = metric_function
        self.optimizer = optimizer
        self.clip_grads = self.config['clip_grads']

        self.logger = logger
        self.checkpoint_dir = join(BASE_DIR, 'checkpoints', run_name)

        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)

        config_filepath = join(self.checkpoint_dir, 'config.json')
        with open(config_filepath, 'w') as config_file:
            json.dump(self.config, config_file)

        self.print_every = self.config['print_every']
        self.save_every = self.config['save_every']

        self.epoch = 0
        self.history = []

        self.start_time = datetime.now()

        self.best_val_metric = None
        self.best_checkpoint_filepath = None

        self.save_format = 'epoch={epoch:0>3}-val_loss={val_loss:<.3}-val_perplexity={val_perplexity}.pth'

        self.log_format = (
            "Epoch: {epoch:>3} "
            "Progress: {progress:<.1%} "
            "Elapsed: {elapsed} "
            "Per second: {per_second:<.1} "
            "Train Loss: {train_loss:<.6} "
            "Val Loss: {val_loss:<.6} "
            "Train Perplexity: {train_perplexity} "
            "Val Perplexity: {val_perplexity} "
            "Learning rate: {current_lr:<.4} "
        )

    def run_epoch(self, dataloader, mode='train'):
        batch_losses = []
        batch_counts = []
        # batch_metrics = []
        for sources, inputs, targets, lengths in tqdm(dataloader):
            sources, inputs, targets = sources.to(self.device), inputs.to(self.device), targets.to(self.device)
            outputs = self.model(sources, inputs)  #, lengths['sources_lengths'], lengths['inputs_lengths']

            batch_loss, batch_count = self.loss_function(outputs, targets, lengths['targets_lengths'])

            if mode == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            batch_losses.append(batch_loss.item())
            batch_counts.append(batch_count)

            # batch_metric, batch_count = self.metric_function(outputs, targets, lengths['targets_lengths'])
            # batch_metrics.append(batch_metric)

            if self.epoch == 0:  # for testing
                return float('inf'), float('inf')

        epoch_loss = sum(batch_losses) / len(dataloader.dataset)
        epoch_metric = float(np.exp(epoch_loss))

        return epoch_loss, epoch_metric

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            self.model.train()

            epoch_start_time = datetime.now()
            train_epoch_loss, train_epoch_metric = self.run_epoch(self.train_dataloader, mode='train')
            epoch_end_time = datetime.now()

            self.model.eval()

            val_epoch_loss, val_epoch_metric = self.run_epoch(self.val_dataloader, mode='val')

            if epoch % self.print_every == 0 and self.logger:
                per_second = len(self.train_dataloader.dataset) / ((epoch_end_time - epoch_start_time).seconds + 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                log_message = self.log_format.format(epoch=epoch,
                                                     progress=epoch / epochs,
                                                     per_second=per_second,
                                                     train_loss=train_epoch_loss,
                                                     val_loss=val_epoch_loss,
                                                     train_perplexity=np.exp(train_epoch_loss),
                                                     val_perplexity=np.exp(val_epoch_loss),
                                                     current_lr=current_lr,
                                                     elapsed=self._elapsed_time()
                                                     )

                self.logger.info(log_message)

            if epoch % self.save_every == 0:
                self._save_model(epoch, train_epoch_loss, val_epoch_loss, train_epoch_metric, val_epoch_metric)

    def _save_model(self, epoch, train_epoch_loss, val_epoch_loss, train_epoch_metric, val_epoch_metric):

        checkpoint_filename = self.save_format.format(
            epoch=epoch,
            val_loss=val_epoch_loss,
            val_perplexity='{:<.3}'.format(val_epoch_metric)
        )

        checkpoint_filepath = join(self.checkpoint_dir, checkpoint_filename)

        save_state = {
            'epoch': epoch,
            'train_loss': train_epoch_loss,
            'train_metric': train_epoch_metric,
            'val_loss': val_epoch_loss,
            'val_metric': val_epoch_metric,
            'checkpoint': checkpoint_filepath,
        }

        if self.epoch > 0:
            torch.save(self.model.state_dict(), checkpoint_filepath)
            self.history.append(save_state)

        if self.best_val_metric is None or self.best_val_metric < val_epoch_metric:
            self.best_val_metric = val_epoch_metric
            self.val_loss_at_best = val_epoch_loss
            self.train_loss_at_best = train_epoch_loss
            self.train_metric_at_best = train_epoch_metric
            self.best_checkpoint_filepath = checkpoint_filepath

        if self.logger:
            self.logger.info("Saved model to {}".format(checkpoint_filepath))
            self.logger.info("Current best model is {}".format(self.best_checkpoint_filepath))

    def _elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed).split('.')[0]  # remove milliseconds


def input_target_collate_fn(batch):
    """merges a list of samples to form a mini-batch.

    Args:
        batch : tuple of inputs and targets. For example,
        ([1, 164, 109, 253, 66, 484, 561, 76, 528, 279, 458],
        [164, 109, 253, 66, 484, 561, 76, 528, 279, 458, 1])
    """
    # indexed_sources = [sources for sources, inputs, targets in batch]
    # indexed_inputs = [inputs for sources, inputs, targets in batch]
    # indexed_targets = [targets for sources, inputs, targets in batch]

    sources_lengths = [len(sources) for sources, inputs, targets in batch]
    inputs_lengths = [len(inputs) for sources, inputs, targets in batch]
    targets_lengths = [len(targets) for sources, inputs, targets in batch]

    sources_max_length = max(sources_lengths)
    inputs_max_length = max(inputs_lengths)
    targets_max_length = max(targets_lengths)

    sources_padded = [sources + [PAD_INDEX] * (sources_max_length - len(sources)) for sources, inputs, targets in batch]
    inputs_padded = [inputs + [PAD_INDEX] * (inputs_max_length - len(inputs)) for sources, inputs, targets in batch]
    targets_padded = [targets + [PAD_INDEX] * (targets_max_length - len(targets)) for sources, inputs, targets in batch]

    sources_tensor = torch.tensor(sources_padded)
    inputs_tensor = torch.tensor(inputs_padded)
    targets_tensor = torch.tensor(targets_padded)

    lengths = {
        'sources_lengths': torch.tensor(sources_lengths),
        'inputs_lengths': torch.tensor(inputs_lengths),
        'targets_lengths': torch.tensor(targets_lengths)
    }

    return sources_tensor, inputs_tensor, targets_tensor, lengths