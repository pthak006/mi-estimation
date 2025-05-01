import math
import numpy as np
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.estimators.neural._critic import UnetMLP
from sklearn import preprocessing
from src.estimators.neural.libs.util import EMA,SNRMMSEPlotter
import json
from MIND_Unet_static_HM import MINDEstimator as MIND_Base
import copy
class Denoiser(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=128, n_layers=3, emb_size=64):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim   
        input_dim = x_dim + y_dim
        hidden_dim = 64 if input_dim <= 10 else 128 if input_dim <= 50 else 256
        self.unet = UnetMLP(dim=input_dim, 
                            init_dim=hidden_dim, 
                            dim_mults=[], 
                            time_dim=hidden_dim, 
                            nb_mod=1,
                            out_dim=x_dim)
            
    def forward(self, x, logsnr, y=None):
        if y is None:
            y = t.zeros(x.shape[0], self.y_dim, device=x.device)
        input_tensor = t.cat([x.flatten(1), y.flatten(1)], dim=1)
        return self.unet(input_tensor, logsnr)




class MINDEstimator(pl.LightningModule):
    def __init__(self, 
             x_shape=(1,), 
             y_shape=(1,),
             learning_rate=1e-4, 
             batch_size=512,
             logsnr_loc=2., 
             logsnr_scale=3., 
             max_n_steps=None, 
             max_epochs=None,
             seed=42, 
             task_name="default", 
             task_gt=-1, 
             test_num=10000, 
             mi_estimation_interval=500,
             update_logsnr_loc_flag=False,
             use_ema=True,
             ema_decay=0.999,
             test_batch_size=1000,
             threshold = 5,
             method= 'noise_prediction'
             ):

        super().__init__()
        self.save_hyperparameters()
        self.d_x = np.prod(x_shape)
        self.d_y = np.prod(y_shape)
        self.h_g = 0.5 * self.d_x * math.log(2 * math.pi * math.e)
        self.left = (-1,) + (1,) * len(x_shape)
        self.mi_estimation_interval = mi_estimation_interval
        
        self.task_name = task_name
        self.logger_name = f"mind_estimator_{task_name}_seed_{seed}_lr_{learning_rate}_method_{method}_moe_location_5.0_scale_4.0"
        self.task_gt = task_gt
        self.test_X = None
        self.test_Y = None
        self.test_num = test_num
        self.logsnr_scale = logsnr_scale
        self.use_ema = use_ema
        if max_n_steps:
            self.logger_name += f"_max_steps_{max_n_steps}"
        if max_epochs:
            self.logger_name += f"_max_epochs_{max_epochs}"
        self.logsnr_loc = t.tensor(logsnr_loc, device=self.device)
        self.update_logsnr_loc_flag = update_logsnr_loc_flag
        self.threshold = threshold




        self.model_low = Denoiser(self.d_x, self.d_y)
        self.model_high = Denoiser(self.d_x, self.d_y)


        self.model_low_ema = EMA(self.model_low, decay=ema_decay) if use_ema else None
        self.model_high_ema = EMA(self.model_high, decay=ema_decay) if use_ema else None

        self.method = method

        self.plotter = None
    def on_before_backward(self, loss: t.Tensor) -> None:
        if self.use_ema:
            self.model_low_ema.update(self.model_low)
            self.model_high_ema.update(self.model_high)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if t.rand(1) < 0.5:
            loss = self.nll(x)            
        else:
            loss = self.nll(x, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.global_step % self.mi_estimation_interval == 0:
            with t.no_grad():
                loss_xy_train = self.nll(x, y)
                loss_x_train = self.nll(x)
                self.logger.experiment.add_scalars(
                    'loss',
                    {
                        'train_loss_unconditional': loss_x_train,
                        'train_loss_conditional': loss_xy_train,
                        'train_loss_mean': (loss_x_train + loss_xy_train) / 2
                    },
                    self.global_step
                )
                
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        x, y = batch
        loss_x = self.nll(x)
        loss_xy = self.nll(x, y)
        mi_estimate, mi_orthogonal = self.estimate_mi_during_training(x, y)
        self.logger.experiment.add_scalars(
                'loss',
                {
                    'estimate': mi_estimate,
                    'orthogonal': mi_orthogonal,
                    'ground_truth': self.task_gt,
                    'validation_loss_unconditional': loss_x,
                    'validation_loss_conditional': loss_xy,
                    'validation_loss_mean': (loss_x + loss_xy) / 2,
                },
                self.global_step)
        return loss_x, loss_xy

    def estimate_mi_during_training(self, x, y):
        self.eval()
        with t.no_grad():
            nll_x = self.nll(x)
            nll_xy = self.nll(x, y)
            mi_estimate = nll_x - nll_xy
            mi_estimate_orthogonal = self.estimate_x_y(x, y)
        self.train()
        return mi_estimate.item(), mi_estimate_orthogonal.item()

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }

    def noisy_channel(self, x, logsnr):
        logsnr = logsnr.view(self.left)
        eps = t.randn_like(x)
        return t.sqrt(t.sigmoid(logsnr)) * x + t.sqrt(t.sigmoid(-logsnr)) * eps, eps

    def mse(self, x, logsnr, y=None):
        # Get noisy input
        z, eps = self.noisy_channel(x, logsnr)
        
        # Set threshold for splitting between high and low SNR
        threshold = self.threshold
        low_mask = logsnr < threshold
        high_mask = logsnr >= threshold
        
        # Initialize total error tensor
        total_error = t.zeros(len(x), device=x.device)
        
        # Process low SNR samples
        if low_mask.any():
            z_low = z[low_mask]
            logsnr_low = logsnr[low_mask]
            eps_low = eps[low_mask]
            y_low = y[low_mask] if y is not None else None
            
            # For low SNR, predict eps directly and compute MSE with true eps
            eps_hat_low = self.model_low(z_low, logsnr_low, y_low)
            error_low = (eps_low - eps_hat_low).flatten(start_dim=1)
            total_error[low_mask] = t.einsum('ij,ij->i', error_low, error_low)
        
        # Process high SNR samples
        if high_mask.any():
            z_high = z[high_mask]
            logsnr_high = logsnr[high_mask]
            x_high = x[high_mask]
            eps_high = eps[high_mask]
            y_high = y[high_mask] if y is not None else None
            
            if self.method == 'noise_prediction':
                eps_hat_high = self.model_high(z_high, logsnr_high, y_high)
                error_high = (eps_high - eps_hat_high).flatten(start_dim=1)
                total_error[high_mask] = t.einsum('ij,ij->i', error_high, error_high)


            #method 1

            if self.method == 'x_prediction':
            # For high SNR, predict x directly
                eps_hat_high = self.model_high(z_high, logsnr_high, y_high)
            
            # Calculate alpha and sigma
                alpha = t.sqrt(t.sigmoid(logsnr_high.view(self.left)))
                sigma = t.sqrt(t.sigmoid(-logsnr_high.view(self.left)))
            
            # Recover x_hat using the correct formula
                x_hat_high = (z_high - sigma * eps_hat_high) / alpha
            
                error_high = (x_high - x_hat_high).flatten(start_dim=1)
            #method 2
            #x_hat_high = self.model_high(z_high, logsnr_high, y_high)
            #error_high = (x_high - x_hat_high).flatten(start_dim=1)

            
            
            total_error[high_mask] = t.einsum('ij,ij->i', error_high, error_high)



        return total_error


    def nll(self, x, y=None):
        logsnr, weights = self.logistic_integrate(len(x))
        mses = self.mse(x, logsnr, y)
        mmse_gap = mses - self.d_x * t.sigmoid(logsnr)
        return self.h_g + 0.5 * (weights * mmse_gap).mean()

    def logistic_integrate(self, npoints, clip=4.):
        loc, scale = self.logsnr_loc, self.logsnr_scale
        loc, scale, clip = map(lambda x: t.tensor(x, device=self.device), [loc, scale, clip])
        ps = t.rand(npoints, device=self.device)
        ps = t.sigmoid(-clip) + (t.sigmoid(clip) - t.sigmoid(-clip)) * ps
        logsnr = loc + scale * (t.log(ps) - t.log(1-ps))
        weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
        return logsnr, weights

    def mse_x_y(self, x, logsnr, y):
        """Modified to use low/high models based on SNR threshold"""
        z, eps = self.noisy_channel(x, logsnr)
        
        # Set threshold for splitting between high and low SNR
        threshold = self.threshold
        low_mask = logsnr < threshold
        high_mask = logsnr >= threshold
        
        # Initialize error tensor
        error = t.zeros_like(x).flatten(1)
        
        # Process low SNR samples
        if low_mask.any():
            z_low = z[low_mask]
            logsnr_low = logsnr[low_mask]
            y_low = y[low_mask] if y is not None else None
            
            eps_hat_x_low = self.model_low(z_low, logsnr_low)
            eps_hat_y_low = self.model_low(z_low, logsnr_low, y_low)
            error[low_mask] = eps_hat_x_low - eps_hat_y_low
        
        # Process high SNR samples
        if high_mask.any():
            z_high = z[high_mask]
            logsnr_high = logsnr[high_mask]
            y_high = y[high_mask] if y is not None else None
            
            eps_hat_x_high = self.model_high(z_high, logsnr_high)
            eps_hat_y_high = self.model_high(z_high, logsnr_high, y_high)
            error[high_mask] = eps_hat_x_high - eps_hat_y_high
        
        return t.einsum('ij,ij->i', error, error)

    def estimate_x_y(self, x, y):
        logsnr, weights = self.logistic_integrate(len(x))
        mses = self.mse_x_y(x, logsnr, y)
        return 0.5 * (weights * mses).mean()
    
    def configure_logger(self, train_sample_num):
        logger = TensorBoardLogger("lightning_logs", name=self.logger_name+"-"+str(train_sample_num))
        logger.log_hyperparams({'train_sample_num': train_sample_num}, {'test_sample_num': self.test_num})
        return logger

    def configure_trainer(self, train_sample_num, **trainer_kwargs):
        #log the hyperparameters of training_sample_num
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints/{self.task_name}',
            filename=f'mind_estimator-{self.logger_name}-{train_sample_num}',
            save_last=True,
            save_top_k=1,
            monitor='train_loss',
            mode='min'
        )
        logger = self.configure_logger(train_sample_num)
        self.plotter = SNRMMSEPlotter(self, self.task_name,logger, self.logger_name)
        trainer_kwargs.update({
            'callbacks': [checkpoint_callback],
            'logger': logger,
            'log_every_n_steps': 10,
    })
       
        if self.hparams.max_n_steps:
            trainer_kwargs['max_steps'] = self.hparams.max_n_steps
        if self.hparams.max_epochs:
            trainer_kwargs['max_epochs'] = self.hparams.max_epochs

        trainer = pl.Trainer(**trainer_kwargs )
        return trainer
    def estimate_mmse_gap(self, x, y, logsnrs, weights):
        self.eval()
        with t.no_grad():
            num_logsnr_steps = len(logsnrs)
            x = t.tensor(x, dtype=t.float32).to(self.device) if not isinstance(x, t.Tensor) else x
            y = t.tensor(y, dtype=t.float32).to(self.device) if not isinstance(y, t.Tensor) else y
            
            N = x.shape[0]
            x_flat = x.reshape(N, -1)
            y_flat = y.reshape(N, -1) if y.dim() > 1 else y.reshape(N, 1)
            
            sort_indices = t.argsort(logsnrs)
            logsnrs = logsnrs[sort_indices]
            weights = weights[sort_indices]
            
            x_repeated = x_flat.unsqueeze(0).expand(num_logsnr_steps, -1, -1)
            y_repeated = y_flat.unsqueeze(0).expand(num_logsnr_steps, -1, -1)
            logsnrs_repeated = logsnrs.unsqueeze(1).expand(-1, N).unsqueeze(-1)
            
            x_reshaped = x_repeated.reshape(-1, x_flat.shape[-1])
            y_reshaped = y_repeated.reshape(-1, y_flat.shape[-1])
            logsnrs_reshaped = logsnrs_repeated.reshape(-1, 1)

            threshold = self.threshold
            low_mask = (logsnrs_reshaped < threshold).squeeze(-1)
            high_mask = (logsnrs_reshaped >= threshold).squeeze(-1)

            unconditional_ehat = t.zeros_like(x_reshaped)
            conditional_ehat = t.zeros_like(x_reshaped)

            z, eps = self.noisy_channel(x_reshaped, logsnrs_reshaped)

            if low_mask.any():
                z_low = z[low_mask]
                logsnr_low = logsnrs_reshaped[low_mask]
                y_low = y_reshaped[low_mask] if y_reshaped is not None else None
                
                eps_hat_x_low = self.model_low(z_low, logsnr_low)
                eps_hat_y_low = self.model_low(z_low, logsnr_low, y_low)
                
                unconditional_ehat[low_mask] = eps_hat_x_low
                conditional_ehat[low_mask] = eps_hat_y_low

            if high_mask.any():
                z_high = z[high_mask]
                logsnr_high = logsnrs_reshaped[high_mask]
                y_high = y_reshaped[high_mask] if y_reshaped is not None else None
                
                eps_hat_x_high = self.model_high(z_high, logsnr_high)
                eps_hat_y_high = self.model_high(z_high, logsnr_high, y_high)
                unconditional_ehat[high_mask] = eps_hat_x_high
                conditional_ehat[high_mask] = eps_hat_y_high

            error_unconditional = t.einsum('ij,ij->i', (unconditional_ehat - eps).flatten(start_dim=1), 
                                        (unconditional_ehat - eps).flatten(start_dim=1))
            error_conditional = t.einsum('ij,ij->i', (conditional_ehat - eps).flatten(start_dim=1),
                                    (conditional_ehat - eps).flatten(start_dim=1))
            
            mses_unconditional = error_unconditional.view(num_logsnr_steps, N).mean(dim=1)
            mses_conditional = error_conditional.view(num_logsnr_steps, N).mean(dim=1)
            unconditional_ehat = unconditional_ehat.view(num_logsnr_steps, N, -1)
            conditional_ehat = conditional_ehat.view(num_logsnr_steps, N, -1)

        return mses_unconditional, mses_conditional, unconditional_ehat, conditional_ehat
    def fit(self, X: np.ndarray, Y: np.ndarray,X_test: np.ndarray, Y_test: np.ndarray):
        train_sample_num = len(X)
        
        dataset = TensorDataset(t.tensor(X, dtype=t.float32), t.tensor(Y, dtype=t.float32))
        train_data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True) 
        validation_dataset = TensorDataset(t.tensor(X_test, dtype=t.float32), t.tensor(Y_test, dtype=t.float32))
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.hparams.test_batch_size, shuffle=False)

        trainer = self.configure_trainer(train_sample_num)
        trainer.fit(model=self, train_dataloaders=train_data_loader,
                val_dataloaders=validation_dataloader)
        return self

    def estimate(self, X, Y, n_samples=1000) -> float:
        self.eval()
        tmp_model_low = self.model_low
        tmp_model_high = self.model_high
        if self.use_ema:
            self.model_low = self.model_low_ema.module
            self.model_high = self.model_high_ema.module

        X = t.tensor(X, dtype=t.float32).to(self.device)
        Y = t.tensor(Y, dtype=t.float32).to(self.device)
        
        with t.no_grad():
            mean_estimate = []
            mean_orthogonal = []
            mean_estimate_new = []
            for _ in range(10):
                nll_x = self.nll(X)
                nll_xy = self.nll(X, Y)
                mi_estimate = nll_x - nll_xy
                mi_estimate_orthogonal = self.estimate_x_y(X, Y)
                mi__estimate_new = 0
                mean_estimate.append(mi_estimate)
                print(mi_estimate)
                mean_orthogonal.append(mi_estimate_orthogonal)
                mean_estimate_new.append(mi__estimate_new)
            mi_estimate = t.stack(mean_estimate).mean()
            mi_estimate_orthogonal = t.stack(mean_orthogonal).mean()
            mean_estimate_new =0
        
        self.model_low = tmp_model_low
        self.model_high = tmp_model_high
        return mi_estimate.item(), mi_estimate_orthogonal.item()
    
    @classmethod
    def load_model(cls, checkpoint_path, **kwargs):
        '''
        To load the model, use
        model = MINDEstimator.load_model(checkpoint_path)
        '''
        model = cls.load_from_checkpoint(checkpoint_path, **kwargs)
        return model
    
    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            checkpoint['model_low_ema_state_dict'] = self.model_low_ema.state_dict()
            checkpoint['model_high_ema_state_dict'] = self.model_high_ema.state_dict()
        if self.plotter is not None:
            checkpoint['plotter_task_name'] = self.plotter.task_name
            checkpoint['plotter_logger_name'] = self.plotter.logger_name
            checkpoint['plotter_num_bins'] = self.plotter.num_bins
    
    def on_load_checkpoint(self, checkpoint):
        if self.use_ema:
            if 'model_low_ema_state_dict' in checkpoint:
                self.model_low_ema.load_state_dict(checkpoint['model_low_ema_state_dict'])
            if 'model_high_ema_state_dict' in checkpoint:
                self.model_high_ema.load_state_dict(checkpoint['model_high_ema_state_dict'])
    
        if 'plotter_task_name' in checkpoint:
            self.plotter = SNRMMSEPlotter(
                self, 
                checkpoint['plotter_task_name'], 
                self.logger, 
                checkpoint['plotter_logger_name'], 
                num_bins=checkpoint['plotter_num_bins']
            )
    @staticmethod
    def logsnr_to_weight(logsnr, loc, scale, clip=4.):
        """
        Convert a given logsnr to its corresponding weight.
        
        Args:
        logsnr (torch.Tensor): The input logsnr value(s)
        loc (float): The location parameter of the logistic distribution
        scale (float): The scale parameter of the logistic distribution
        clip (float): The clipping value, default is 4.0
        
        Returns:
        torch.Tensor: The corresponding weight(s) for the input logsnr
        """
        # Ensure all inputs are tensors on the same device as logsnr
        loc = t.tensor(loc, device=logsnr.device)
        scale = t.tensor(scale, device=logsnr.device)
        clip = t.tensor(clip, device=logsnr.device)
        
        # Calculate the weight using the formula from the original function
        weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
        
        return weights
    def mse_x_y_new(self, x, logsnr, y):
        z, eps = self.noisy_channel(x, logsnr)
        eps_hat_x = self.model(z, logsnr)
        eps_hat_y = self.model(z, logsnr, y)
        error = (eps_hat_y*(eps_hat_y-eps_hat_x)).flatten(start_dim=1)
        #get sum
        error = error.sum(dim=1)
        return error

    def estimate_x_y_new(self, x, y):
        logsnr, weights = self.logistic_integrate(len(x))
        mses = self.mse_x_y_new(x, logsnr, y)
        return 0.5 * (weights * mses).mean()

if __name__ == '__main__':
    import bmi
    
    task_list = list(bmi.benchmark.BENCHMARK_TASKS.keys())
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--update_logsnr_loc_flag', type=bool, default=False)
    parser.add_argument('--train_sample_num', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preprocess', type=str, default='rescale')
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--load_ckpt', type=bool, default=False)
    parser.add_argument('--copy_base_model', type=bool, default=True)
    parser.add_argument('--method', type=str, default='x_prediction')

    arg = parser.parse_args()
    start = arg.start
    end = arg.end
    seed = arg.seed
    if seed is not None:
            pl.seed_everything(seed, workers=True)
    strength = [200,2000]
    task_list  = []
    dim_list = [3]
    for dim in dim_list:
        for s in strength:
                task = bmi.benchmark.tasks.task_multinormal_sparse(dim_x=dim, dim_y=dim, strength=s)
                half_cube_task = bmi.benchmark.tasks.transform_half_cube_task(task)
                spiral_cube_task = bmi.benchmark.tasks.transform_spiral_task(task)
                #change task_name
                task_name = f"multinormal_sparse_{s}_dim_{dim}"
                half_cube_task_name = f"half_cube_multinormal_sparse_{s}_dim_{dim}"
                spiral_cube_task_name = f"spiral_multinormal_sparse_{s}_dim_{dim}"
                half_cube_task.task_name = half_cube_task_name
                spiral_cube_task.task_name = spiral_cube_task_name
                task.task_name = task_name
                task_list.append(half_cube_task)
                task_list.append(spiral_cube_task)
                task_list.append(task)
    for task in task_list[start:end]:
        task_name = task.task_name
        train_sample_num = arg.train_sample_num
        test_sample_num = 10000
        batch_size = 256
        seed = arg.seed
        mi_estimation_interval = 500
        update_logsnr_loc_flag = arg.update_logsnr_loc_flag
        file_name = f'results_scale_{train_sample_num}_HM_static.json'
        flag = False
        if task.dim_x <= 5:
            max_epochs = 500
            lr = 1e-3
            batch_size = 128
        elif task.dim_x > 5:
            max_epochs = 750
            lr = 2e-3
            batch_size = 256
        #judge if data already exists
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                results = json.load(f)
                for result in results:
                    if result["task"] == task.name and result["seed"] == arg.seed and result["max_epochs"] == max_epochs and result["preprocessing"] == arg.preprocess:
                        #print(f"Task {task.name} with seed {arg.seed} already evaluated")
                        flag = False
        if flag:
            print(f"Task {task.name} with seed {arg.seed} already evaluated")
            continue
            

        X, Y = task.sample(train_sample_num+test_sample_num, seed=seed)
        X, Y = X.__array__(), Y.__array__()
        
        X_test, Y_test = X[train_sample_num:train_sample_num+test_sample_num], Y[train_sample_num:train_sample_num+test_sample_num]
        X, Y = X[:train_sample_num], Y[:train_sample_num]

        if arg.preprocess == 'rescale':
            scaler_X = preprocessing.StandardScaler(copy=True)
            X = scaler_X.fit_transform(X)
            X_test = scaler_X.transform(X_test)
            
            # Scale target variable
            scaler_Y = preprocessing.StandardScaler(copy=True)
            Y = scaler_Y.fit_transform(Y)
            Y_test = scaler_Y.transform(Y_test)
        
        
        
        X_test, Y_test = X_test.__array__(), Y_test.__array__()
        #find location
        def get_location_scale_in_file(file_name,task_name):
            with open(file_name, 'r') as f:
                results = json.load(f)
                for result in results:
                    if result["task"] == task_name:
                        return result["logsnr_loc"],result["logsnr_scale"]
            return None,None
        #load_file_name = "results_multi_seed_moe.json"
        #dynamic_location,dynamic_scale = get_location_scale_in_file(load_file_name,task.name)
        diffusion_mi = MINDEstimator(
            x_shape=(task.dim_x,), 
            y_shape=(task.dim_y,), 
            learning_rate=lr, 
            batch_size=batch_size,
            max_epochs=max_epochs,
            seed=seed, 
            task_name=task_name,
            mi_estimation_interval=mi_estimation_interval,
            task_gt=task.mutual_information,
            update_logsnr_loc_flag=update_logsnr_loc_flag,
            use_ema=arg.use_ema,
            method=arg.method,

        )
        ckpt_path = f'checkpoints/{task_name}/mind_estimator-{diffusion_mi.logger_name}-{train_sample_num}.ckpt'
        if arg.load_ckpt:
            diffusion_mi = MINDEstimator.load_model(checkpoint_path=ckpt_path)
        elif arg.copy_base_model:
            mind_base = MIND_Base(
            x_shape=(task.dim_x,), 
            y_shape=(task.dim_y,), 
            learning_rate=lr, 
            batch_size=batch_size,
            max_epochs=max_epochs,
            seed=seed, 
            task_name=task_name,
            mi_estimation_interval=mi_estimation_interval,
            task_gt=task.mutual_information,
            update_logsnr_loc_flag=update_logsnr_loc_flag,
            use_ema=arg.use_ema,
            logsnr_loc =5.0 ,
            logsnr_scale = 4.0,
            )
            base_model_ckpt_path = f'checkpoints/{task_name}/mind_estimator-{mind_base.logger_name}-{train_sample_num}.ckpt'
            base_model_diffusion = MIND_Base.load_model(checkpoint_path=base_model_ckpt_path)
            #deep copy the model
            diffusion_mi.model_low = copy.deepcopy(base_model_diffusion.model)
            diffusion_mi.model_high = copy.deepcopy(base_model_diffusion.model)
            diffusion_mi.model_low_ema = copy.deepcopy(base_model_diffusion.model_ema)
            diffusion_mi.model_high_ema = copy.deepcopy(base_model_diffusion.model_ema)

            diffusion_mi.fit(X, Y, X_test, Y_test)
            diffusion_mi.trainer.save_checkpoint(ckpt_path)

            

       
        else:

            diffusion_mi.fit(X, Y, X_test, Y_test)
            diffusion_mi.trainer.save_checkpoint(ckpt_path)
        #save model
        

        mi_estimate, mi_orthogonal = diffusion_mi.estimate(X_test, Y_test)
        mi_estimate_new= 0
        diffusion_mi.plotter.plot_improved_snr_mse(t.tensor(X_test), t.tensor(Y_test), gt_mi=task.mutual_information, tag='test')
    
        import json

        result_dict = {
            "task": task.name,
            "gt_mi": task.mutual_information,
            "mi_estimate": mi_estimate,
            "learning_rate": lr,
            "mi_estimate_orthogonal": mi_orthogonal,
            "estimator": diffusion_mi.__class__.__name__,
            "seed": seed,
            "batch_size": batch_size,
            "train_sample_num": arg.train_sample_num,
            "test_sample_num": test_sample_num,
            'log_snr_dynamic': update_logsnr_loc_flag,
            "max_epochs": max_epochs,
            "max_steps": None,
            "hidden_dim": 64 if task.dim_x <= 10 else 128 if task.dim_x <= 50 else 256,
            "time_emb_size": 64 if task.dim_x <= 10 else 128 if task.dim_x <= 50 else 256,
            "n_layers": None,
            "preprocessing": arg.preprocess,
            "use_ema": arg.use_ema,
            "mi_new":mi_estimate_new
            }
        def append_results(result_dict, filename):
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    results = json.load(f)
            else:
                results = []
            results.append(result_dict)
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
        append_results(result_dict, file_name)