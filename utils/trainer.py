import contextlib
import functools
import logging
import os

import torch
from tqdm import tqdm
from transformers import WEIGHTS_NAME, AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
import wandb
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, model, dataloader, num_train_steps, step_callback=None):
        self.args = args
        self.writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard/train"))
        self.model = model
        self.dataloader = dataloader
        self.num_train_steps = num_train_steps
        self.step_callback = step_callback

        self.optimizer = self._create_optimizer(model)
        self.scheduler = self._create_scheduler(self.optimizer)

    def train(self):
        model = self.model
        optimizer = self.optimizer

        if self.args.fp16:
            from apex import amp

            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level=self.args.fp16_opt_level,
                min_loss_scale=self.args.fp16_min_loss_scale,
                max_loss_scale=self.args.fp16_max_loss_scale,
            )

        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        epoch = 0
        global_step = 0
        tr_loss = 0.0

        num_workers = torch.cuda.device_count()

        def maybe_no_sync(step):
            if (hasattr(model, "no_sync") and num_workers > 1 and (step + 1) % self.args.gradient_accumulation_steps != 0):
                return model.no_sync()
            else:
                return contextlib.ExitStack()

        model.train()

        with tqdm(total=self.num_train_steps, disable=self.args.local_rank not in (-1, 0)) as pbar:
            while True:
                for step, batch in enumerate(self.dataloader):
                    inputs = {k: v.to(self.args.device) for k, v in self._create_model_arguments(batch).items()}
                    outputs = model(**inputs)
                    loss = outputs[0]
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    with maybe_no_sync(step):
                        if self.args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                    tr_loss += loss.item()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        if self.args.max_grad_norm != 0.0:
                            if self.args.fp16:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        self.optimizer.step()
                        self.scheduler.step()
                        model.zero_grad()

                        pbar.set_description("epoch: %d loss: %.7f" % (epoch, loss.item()))
                        pbar.update()
                        global_step += 1

                        self.writer.add_scalar('loss', loss.item(), global_step=global_step)
                        self.writer.add_scalar('lr', self.scheduler.get_lr()[0])
                        wandb.log({"step": global_step, "loss": loss.item()})
                        wandb.log({"step": global_step, "lr": self.scheduler.get_lr()[0]})

                        if self.step_callback is not None:
                            self.step_callback(model, global_step)

                        if (self.args.local_rank in (-1, 0) and self.args.output_dir and self.args.save_steps > 0 and global_step % self.args.save_steps == 0):
                            output_dir = os.path.join(self.args.output_dir, "checkpoint-{}".format(global_step))

                            if not os.path.exists(output_dir):
                                os.mkdir(output_dir)

                            if hasattr(model, "module"):
                                torch.save(model.module.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
                            else:
                                torch.save(model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

                        if global_step == self.num_train_steps:
                            break

                if global_step == self.num_train_steps:
                    break
                epoch += 1

        logger.info("global_step = %s, average loss = %s", global_step, tr_loss / global_step)

        return model, global_step, tr_loss / global_step

    def _create_optimizer(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(
            optimizer_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_eps,
            betas=(self.args.adam_b1, self.args.adam_b2),
            correct_bias=self.args.adam_correct_bias,
        )

    def _create_scheduler(self, optimizer):
        warmup_steps = int(self.num_train_steps * self.args.warmup_proportion)
        if self.args.lr_schedule == "warmup_linear":
            return get_linear_schedule_with_warmup(optimizer, warmup_steps, self.num_train_steps)
        if self.args.lr_schedule == "warmup_constant":
            return get_constant_schedule_with_warmup(optimizer, warmup_steps)

        raise RuntimeError("Unsupported scheduler: " + self.args.lr_schedule)

    def _create_model_arguments(self, batch):
        return batch
