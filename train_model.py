import datetime
import pytz
import os
import argparse
import importlib
import util
import util.data
import util.data.functions
import numpy as np
import torch
import pdb
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help='size of each batch')
parser.add_argument('--buffer_size', type=int, default=5, help='number of images to cache in memory')
parser.add_argument('--path_data', default='data', help='path to data directory')
parser.add_argument('--data_provider_module', default='multifiledataprovider', help='data provider class')
parser.add_argument('--data_set_module', default='dataset', help='data set class')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
parser.add_argument('--iter_checkpoint', type=int, default=500, help='iterations between saving log/model checkpoints')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--model_module', default='ttf_model', help='name of the model module')
parser.add_argument('--n_iter', type=int, default=500, help='number of training iterations')
parser.add_argument('--nn_module', default='ttf_v8_nn', help='name of neural network module')
parser.add_argument('--replace_interval', type=int, default=-1, help='iterations between replacements of images in cache')
parser.add_argument('--resume_path', help='path to saved model to resume training')
parser.add_argument('--run_name', required=True, help='name of run')
parser.add_argument('--path_save_parent', default='saved_models', help='base directory for saved models')
parser.add_argument('--seed', type=int, default=666, help='random seed')
opts = parser.parse_args()

model_module = importlib.import_module('model_modules.' + opts.model_module)
data_provider_module = importlib.import_module('util.data.' + opts.data_provider_module)
data_set_module = importlib.import_module('util.data.' + opts.data_set_module)

def train_model(model, data):
    path_run_dir = os.path.join(opts.path_save_parent, opts.run_name)
    if not os.path.exists(path_run_dir):
        os.makedirs(path_run_dir)

    fo = open(os.path.join(path_run_dir, 'run.log'), 'w')
    print(get_start_time(), file=fo)
    print(vars(opts), file=fo)
        
    logger = util.SimpleLogger(('num_iter', 'loss', 'sources'),
                               'num_iter: {:4d} | loss: {:.4f} | sources: {:s}')

    start = time.time()
    for i in range(opts.n_iter):
        x, y = data.get_batch()
        loss = model.do_train_iter(x, y)
        str_out = logger.add((
            i,
            loss,
            data.last_sources
        ))
        print(str_out, file=fo)
        if ((i + 1) % opts.iter_checkpoint == 0) or ((i + 1) == opts.n_iter):
            logger.save_csv(os.path.join(path_run_dir, 'log.csv'))
            model.save_checkpoint(os.path.join(path_run_dir, 'model.p'))
            
    t_elapsed = time.time() - start
    print('total training time: {:.1f} s'.format(t_elapsed), file=fo)
    fo.close()

def get_start_time():
    now_dt = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific'))
    return now_dt.strftime('%y-%m-%d %H:%M:%S')

def main():
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    main_gpu_id = opts.gpu_ids if isinstance(opts.gpu_ids, int) else opts.gpu_ids[0]
    torch.cuda.set_device(main_gpu_id)
    print('main GPU ID:', torch.cuda.current_device())

    if opts.resume_path is None:
        model = model_module.Model(lr=opts.lr, nn_module=opts.nn_module,
                                   gpu_ids=opts.gpu_ids
        )
    else:
        model = model_module.Model(load_path=opts.resume_path,
                                   gpu_ids=opts.gpu_ids
        )
    print(model)
    
    dataset = util.data.functions.load_dataset(opts.path_data)
    print(dataset)
    
    data_train = data_provider_module.DataProvider(
        dataset,
        buffer_size=opts.buffer_size,
        batch_size=opts.batch_size,
        replace_interval=opts.replace_interval,
    )
    train_model(model, data_train)

    
if __name__ == '__main__':
    main()
