from genericpath import exists
import os
import logging

import mlflow
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.metrics import AverageMeter, MetricCalculator

logging.basicConfig(level=logging.INFO)

torch.manual_seed(42)
np.random.seed(42)


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = self.model.criterion
        self.optimizer = self.model.optimizer
        self.scheduler = self.model.scheduler
        self.accum_iter = config.strat.accum_iter
        os.makedirs(self.config.out_path, exist_ok=True)

    def run_epoch(self, model, loader, train=True):
        if train:
            model.train()
            metrics = AverageMeter()
        else:
            model.eval()
            metrics = MetricCalculator()

        tk1 = tqdm(loader, total=int(len(loader)))

        self.optimizer.zero_grad()
        batch_idx = 0
        for data in tk1:
            if train:
                loss = self.model.iteration(data, train)
                if isinstance(loss, dict):
                    loss["loss"] /= self.accum_iter
                    loss["loss"].backward()
                    loss["loss"] *= self.accum_iter
                else:
                    loss /= self.accum_iter
                    loss.backward()
                    loss *= self.accum_iter
                if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if (
                    "scheduler_batch_step" in self.config.strat
                    and self.config.strat.scheduler_batch_step
                ):
                    self.scheduler.step()
            else:
                with torch.no_grad():
                    output = self.model.iteration(data, train)

            batch_idx += 1
            labels = data["target"] if isinstance(data, dict) else data[1]

            if train:
                metrics.add(loss, labels.shape[0])

            if not train:
                metrics.add(output)

            if batch_idx >= self.config.strat.epoch_len:
                break

        result = metrics.get()
        return result

    def run(self, train_loader, valid_loader, loaders=[]):
        experiment_id = mlflow.set_experiment(self.config.experiment_name)
        with mlflow.start_run(experiment_id=experiment_id):
            # usefull_config = {
            #     key: self.config[key] for key in self.config if "transform" not in key
            # }
            # mlflow.log_params(usefull_config)
            mlflow.log_param("criterion", self.criterion)

            num_epochs = self.config.strat.epochs
            for epoch in range(num_epochs):
                logging.info(f"Epoch: {epoch}.    Train:")

                train_res = self.run_epoch(
                    self.model, train_loader, train=True
                )
                self._print_result("Train", train_res)
                logging.info("Validation:")
                val_res = self.run_epoch(
                    self.model, valid_loader, train=False
                )
                self._print_result("Val", val_res)

                if self.config.strat.scheduler_step:
                    self.scheduler.step(val_res["accuracy"])

                self._save_model(epoch, train_res, val_res)

                self._log_to_mlflow(epoch, train_res)
                self._log_to_mlflow(epoch, val_res, "val_")

                for i, eloader in enumerate(loaders):
                    eval_res = self.evaluate(eloader)
                    self._log_to_mlflow(epoch, eval_res, f"val_{i + 1}_")

    def evaluate(self, loader):
        eval_res = self.run_epoch(self.model, loader, train=False)
        self._print_result("Eval", eval_res)
        return eval_res

    def _log_to_mlflow(self, epoch, result, prefix=""):
        for key in result:
            if isinstance(result[key], dict):
                nested_prefix = prefix + key + "_"
                self._log_to_mlflow(epoch, result[key], nested_prefix)
            if isinstance(result[key], list):
                pass
            else:
                mlflow.log_metric(prefix + key, result[key], step=epoch)
        
        if "frr" in result:
            frr, far, thresh = result["frr"], result["far"], result["thresh"]
            for idx in range(len(frr)):
                seuil = int(round(thresh[idx] * 10000, 0))
                mlflow.log_metrics({f"FRR_{epoch}": frr[idx], f"FAR_{epoch}": far[idx]}, step=seuil)

    def _print_result(self, stage, result_dict):
        result = "".join(f"{key}:   {value}   " for key, value in result_dict.items() if not (isinstance(value, dict) or isinstance(value, list)))
        logging.info(f"{stage}: " + result)

        for key in result_dict:
            if isinstance(result_dict[key], dict):
                self._print_result(stage + "_" + key, result_dict[key])

    def _save_model(self, epoch, train_res, val_res):
        train_loss = train_res["loss"]
        val_loss = val_res["accuracy"]
        path = f"{self.config.out_path}/{self.config.experiment_name}_{epoch}_{train_loss}_{val_loss}.pth"
        self.model.save(path)
