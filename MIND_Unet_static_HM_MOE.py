from MIND_Unet_static_HM import MINDEstimator
import numpy as np
import torch as t
import pytorch_lightning as pl
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from sklearn import preprocessing
import json
from tqdm import tqdm

def logistic_integrate(logsnr_loc, logsnr_scale, npoints, clip=4., device='cuda'):
    # Directly convert to tensors
    loc = t.tensor(logsnr_loc, device=device)
    scale = t.tensor(logsnr_scale, device=device)
    clip = t.tensor(clip, device=device)
    # Generate random points and precompute sigmoid of clip
    ps = t.rand(npoints, device=device)
    ps = t.sigmoid(-clip) + (t.sigmoid(clip) - t.sigmoid(-clip)) * ps
    logsnr = loc + scale * (t.log(ps) - t.log(1-ps))
    weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
    return logsnr, weights


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
    parser.add_argument('--load_ckpt', type=bool, default=True)

    arg = parser.parse_args()
    start = arg.start
    end = arg.end
    seed = arg.seed
    if seed is not None:
            pl.seed_everything(seed, workers=True)
    strength = [200, 650, 1100, 1550, 2000]
    task_list  = []
    dim_list = [3,5,10,20,50]
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
        file_name = f'results_scale_{train_sample_num}_HM_static_MOE.json'
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
        )
        ckpt_path = f'checkpoints/{task_name}/mind_estimator-{diffusion_mi.logger_name}-{train_sample_num}.ckpt'
        
        
        if arg.load_ckpt:

            ckpt_path = f'checkpoints/{task_name}/mind_estimator-{diffusion_mi.logger_name}-{train_sample_num}.ckpt'

            #seed_0 to 9
            path_list = os.listdir(f'checkpoints/{task_name}')
            num_point = 100
            import re
            seed_pattern = re.compile(r'seed_[0-9](?![0-9])')


            filter_paths = []
            for path in path_list:
                if "last"  not in path and seed_pattern.search(path):
                    filter_paths.append( f'checkpoints/{task_name}/{path}')

            print(f"filter_paths:{filter_paths}")
            logsnr, weights = logistic_integrate(2, 3, num_point)

            logsnr, sort_indices = t.sort(logsnr)
            weights = weights[sort_indices]
            #send logsnr and weights to device
            

            mmse_result = {}
            mi_result = {}
            uncondtional_list = []
            conditional_list = []
            eps_hat_unconditional_list = []
            eps_hat_conditional_list = []
            first_flag = True
            orthogonal_list =  []
            for seed,ckpt_path in enumerate(filter_paths):
                print(f"ckpt_path:{ckpt_path}")
                diffusion_mi_0 = MINDEstimator.load_model(checkpoint_path=ckpt_path)
                logsnr = logsnr.to(diffusion_mi_0.device)
                weights = weights.to(diffusion_mi_0.device)
                
                mses_unconditional_0,mses_conditional_0,conditional_ehat,uncondtional_ehat =diffusion_mi_0.estimate_mmse_gap(X_test, Y_test, logsnr, weights)
                location,scale = diffusion_mi_0.estimate_location_scale(X_test, Y_test)
                mi_result[f"{seed}_location"] = location
                mi_result[f"{seed}_scale"] = scale



                mmse_result[f"{seed}_unconditional"] = mses_unconditional_0.detach().cpu()
                mmse_result[f"{seed}_conditional"] = mses_conditional_0.detach().cpu()
                uncondtional_list.append(mses_unconditional_0)
                conditional_list.append(mses_conditional_0)
                eps_hat_conditional_list.append(conditional_ehat)
                eps_hat_unconditional_list.append(uncondtional_ehat)
                uncondtional_ehat = uncondtional_ehat.view(num_point*test_sample_num, -1)
                conditional_ehat = conditional_ehat.view(num_point*test_sample_num, -1)
                error = (uncondtional_ehat - conditional_ehat).flatten(start_dim=1)
                error = t.einsum('ij,ij->i', error, error)
                error = error.view(num_point, test_sample_num)
                error = error.mean(dim=1)
                orthogonal_result = 0.5*((error*weights).mean())
                mi_estimate_0 = ((mses_unconditional_0-mses_conditional_0)*weights*0.5).mean()
                print(f"mi_estimate_0:{mi_estimate_0}")
                print(f"orthogonal_result:{orthogonal_result}")
                mi_result[f"{seed}_estimate"] = mi_estimate_0.item()
                orthogonal_list.append(orthogonal_result.item())


                
            
            
            #get min of uncoditional and conditional
            min_unconditional,min_conditional_index = t.min(t.stack(uncondtional_list),dim=0)
            min_conditional,min_conditiontional_index = t.min(t.stack(conditional_list),dim=0)

            def optimized_stack_and_index(tensor_list, index):
                stacked_tensor = t.stack(tensor_list)
                print(f"stacked_tensor:{stacked_tensor.shape}")
                result = stacked_tensor[index, t.arange(stacked_tensor.shape[1])]
                return result

            min_uncondtional_eps_hat = optimized_stack_and_index(eps_hat_conditional_list, min_conditiontional_index)
            min_conditional_eps_hat = optimized_stack_and_index(eps_hat_unconditional_list, min_conditional_index)

            min_uncondtional_eps_hat = min_uncondtional_eps_hat.view(num_point*test_sample_num, -1)
            min_conditional_eps_hat = min_conditional_eps_hat.view(num_point*test_sample_num, -1)
            error = (min_uncondtional_eps_hat - min_conditional_eps_hat).flatten(start_dim=1)
            error = t.einsum('ij,ij->i', error, error)
            error = error.view(num_point, test_sample_num)
            error = error.mean(dim=1)
            orthogonal_result = (error*weights).mean()*0.5


            mmse_result["min_unconditional"] = min_unconditional.detach().cpu()
            mmse_result["min_conditional"] = min_conditional.detach().cpu()
            
            term = min_unconditional-min_conditional
            term = t.maximum(term, t.zeros_like(term))
            mi_result['mean_mi'] = np.mean([mi_result[key] for key in mi_result if "orthogonal" not in key])
            mi_result['mean_orthogonal'] = np.mean(orthogonal_list)
            print(f"mean_mi:{mi_result['mean_mi']}")
            print(f"mean_orthogonal:{mi_result['mean_orthogonal']}")
            mi_result["moe"] = (0.5*(term)*weights).mean().detach().cpu().numpy()
            mi_result["moe_orthogonal"] = orthogonal_result.detach().cpu()
            print(f"moe:{mi_result['moe']}")
            print(f"moe_orthogonal:{mi_result['moe_orthogonal']}")
            mi_result["mean_location"] = np.mean([mi_result[key] for key in mi_result if "location" in key])
            mi_result["mean_scale"] = np.mean([mi_result[key] for key in mi_result if "scale" in key])
        

        #mi_estimate, mi_orthogonal = diffusion_mi.estimate(X_test, Y_test)
        #mi_estimate_new= 0
        #diffusion_mi.plotter.plot_improved_snr_mse(t.tensor(X_test), t.tensor(Y_test), gt_mi=task.mutual_information, tag='test')
    
        import json

        result_dict = {
            "task": task.name,
            "gt_mi": task.mutual_information,
            "learning_rate": lr,
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
            "moe": mi_result["moe"].item(),
            "mean_mi": mi_result['mean_mi'].item(),
            "moe_orthogonal": mi_result["moe_orthogonal"].item(),
            "mean_orthogonal": mi_result['mean_orthogonal'].item(),
            "mean_location": mi_result["mean_location"].item(),
            "mean_scale": mi_result["mean_scale"].item(),
            "seed_0_location": mi_result["0_location"].item(),
            "seed_0_scale": mi_result["0_scale"].item(),
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
