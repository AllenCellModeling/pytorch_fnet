import datetime
import pytz
import os
import argparse
import importlib
import matplotlib.pyplot as plt
import util
import util.data
import numpy as np
import torch
import pdb
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help='size of each batch')
parser.add_argument('--buffer_size', type=int, default=5, help='number of images to cache in memory')
parser.add_argument('--data_path', default='data', help='path to data directory')
parser.add_argument('--data_provider_module', default='multifiledataprovider', help='data provider class')
parser.add_argument('--data_set_module', default='dataset', help='data set class')
parser.add_argument('--dont_init_weights', action='store_true', help='do not init nn weights')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
parser.add_argument('--iter_checkpoint', type=int, default=500, help='iterations between saving log/model checkpoints')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--model_module', default='ttf_model', help='name of the model module')
parser.add_argument('--n_iter', type=int, default=500, help='number of training iterations')
parser.add_argument('--nn_module', default='nosigmoid_nn', help='name of neural network module')
parser.add_argument('--replace_interval', type=int, default=-1, help='iterations between replacements of images in cache')
parser.add_argument('--resume_path', help='path to saved model to resume training')
parser.add_argument('--run_name', help='name of run')
parser.add_argument('--save_dir', default='saved_models', help='save directory for trained model')
parser.add_argument('--seed', type=int, default=666, help='random seed')
opts = parser.parse_args()

model_module = importlib.import_module('model_modules.' + opts.model_module)
data_provider_module = importlib.import_module('util.data.' + opts.data_provider_module)
data_set_module = importlib.import_module('util.data.' + opts.data_set_module)

def train(model, data, logger):
    start = time.time()
    for i, batch in enumerate(data):
        x, y = batch
        loss = model.do_train_iter(x, y)
        logger.add((
            i,
            loss,
            data.last_sources
        ))
        if (i + 1) % opts.iter_checkpoint == 0:
            logger.save_csv()
            model.save_checkpoint(os.path.join(opts.save_dir, logger.logger_name + '.p'))
            # TODO add testing with current model
            
    t_elapsed = time.time() - start
    print('***** Training Time *****')
    print('total:', t_elapsed)
    print()

def create_run_folder():
    pass
    
def gen_run_name():
    now_dt = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific'))
    run_name = now_dt.strftime('run_%y%m%d_%H%M%S')
    return run_name

def main():
    # set seeds for randomizers
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    main_gpu_id = opts.gpu_ids if isinstance(opts.gpu_ids, int) else opts.gpu_ids[0]
    torch.cuda.set_device(main_gpu_id)
    print('main GPU ID:', torch.cuda.current_device())

    run_name = opts.run_name
    if run_name is None:
        run_name = gen_run_name()
    logger = util.SimpleLogger(('num_iter', 'loss', 'sources'),
                               'num_iter: %4d | loss: %.6f | sources: %s',
                               logger_name=run_name)
    

    # instatiate/load model
    if opts.resume_path is None:
        model = model_module.Model(lr=opts.lr, nn_module=opts.nn_module,
                                   init_weights=(not opts.dont_init_weights),
                                   gpu_ids=opts.gpu_ids
        )
    else:
        model = model_module.Model(load_path=opts.resume_path,
                                   gpu_ids=opts.gpu_ids
        )
    print(model)
    
    # get training dataset
    dataset = data_set_module.DataSet(opts.data_path, train_select=True)
    print('DEBUG: data_set_module', data_set_module)
    print(dataset)

    transforms = None
    data_train = data_provider_module.DataProvider(
        dataset,
        buffer_size=opts.buffer_size,
        n_iter=opts.n_iter,
        batch_size=opts.batch_size,
        replace_interval=opts.replace_interval,
        transforms=transforms
    )
    
    train(model, data_train, logger)
        
    # display logger data
    logger.save_csv()
    
    # save model
    model.save_checkpoint(os.path.join(opts.save_dir, run_name + '.p'))
        
        
if __name__ == '__main__':
    main()
