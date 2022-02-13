# Training of a CycleGAN model

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tt
import argparse
import contextlib
import os
import glob
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = False

from data.dataset import CycleGanDataset
from model.discriminator import Discriminator
from model.generator import Generator
from utils.mse_loss_redefined import MSELossRedefined
from utils.LR_Function import LRPolicy
from utils.weights_initialization import weights_init
from utils.buffer import BufferImages
from utils.denormalization import denorm
from utils.save_and_load import save_checkpoint_state, load_checkpoint_state


def arguments_initialization():
    parser = argparse.ArgumentParser(description="GAN project (Deep Learning School; 2022)")

    parser.add_argument('--path_to_data', type=str, default="datasets/current_dataset/", help="Path to data (data for each class should be separated to train and test subfolders, e.g. (trainX,trainY,testX,testY) or (trainA,trainB,testA,testB)).")
    parser.add_argument('--path_save_model', type=str, default="checkpoints/", help="Path to save/load checkpoints.")
    parser.add_argument('--path_save_images', type=str, default="saved_images/", help="Path to save preliminary images generated from train dataset during training procedure.")
    
    parser.add_argument("--image_size", type=int, default=256, help="Size of images ( image's height is equal to width).")
    parser.add_argument("--batch_size", type=int, default=1, help="Size of batch.")
    parser.add_argument("--num_workers", type=int, default=2, help="Num_workers parameter of DataLoader.")
    parser.add_argument('--use_shuffle', action='store_true', default=False, help="Train data is shuffled if set to True.")
    parser.add_argument('--use_amp', action='store_true', default=False, help="If True then Automatic Mixed Precision (AMP) would be used, except for cpu and p100 gpu cases.")    

    parser.add_argument('--make_load', action='store_true', default=False, help="Load pretrained model if set to True.")
    parser.add_argument("--load_epoch", type=int, default=1, help="Epoch to load pretrained model (note, that checkpoint for [--load_epoch] epoch should be in location [--path_save_model]). Irrelevant parameter if [--make_load] parameter set to False.")
    
    parser.add_argument("--epochs", type=int, default=300, help="Total number of epochs.")
    parser.add_argument("--lr_consts", type=float, default=0.0002, help="Initial learning rate (it linearly decays to zero after [--decay_epoch] epochs).")
    parser.add_argument("--decay_epoch", type=int, default=200, help="Epoch after which learning rate parameter starts linearly decay.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Parameter beta1 in Adam optimizer.")    
    parser.add_argument("--lambda_xy", type=float, default=10.0, help="Weight of the cycle consistency loss (lambda_x=lambda_y).")
    parser.add_argument("--lambda_id", type=float, default=0.05, help="Weight of the identity loss.")
    
    parser.add_argument('--use_buffer', action='store_true', default=False, help="Image buffer is used if True.")
    parser.add_argument("--buffer_depth", type=int, default=50, help="Maximum depth of the buffer.")
    
    parser.add_argument("--save_model_interval", type=int, default=1, help="Epoch interval between checkpoints.")
    parser.add_argument("--save_images_interval", type=int, default=50, help="Interval for saving preliminary images generated from train dataset during an epoch loop.")
    
    return parser.parse_args()     


def ifP100():
    return 'P100' in torch.cuda.get_device_name(0)


def fit(model, criterions, data_loader, is_amp):
    
    torch.cuda.empty_cache()

    if args.make_load:
        load_epoch = args.load_epoch
    else:
        load_epoch = 0

    optimizers = {
        "optimizer_dis": torch.optim.Adam(
            list(model["dis_x"].parameters()) + list(model["dis_y"].parameters()), 
            lr=args.lr_consts, betas=(args.beta1, 0.999)),

        "optimizer_gen": torch.optim.Adam(
            list(model["gen_x"].parameters()) + list(model["gen_y"].parameters()), 
            lr=args.lr_consts, betas=(args.beta1, 0.999))
    }
     
    fake_x_buffer = BufferImages(depth=args.buffer_depth, use_buffer=args.use_buffer)
    fake_y_buffer = BufferImages(depth=args.buffer_depth, use_buffer=args.use_buffer)

    schedulers = {
        "scheduler_dis": torch.optim.lr_scheduler.LambdaLR(
            optimizers["optimizer_dis"], 
            lr_lambda = LRPolicy(args.epochs, args.decay_epoch, load_epoch)),

        "scheduler_gen": torch.optim.lr_scheduler.LambdaLR(
            optimizers["optimizer_gen"], 
            lr_lambda = LRPolicy(args.epochs, args.decay_epoch, load_epoch))
        } 

    if args.make_load:
        model, optimizers, schedulers = load_checkpoint_state(load_epoch, model,
                                                              optimizers, 
                                                              schedulers, 
                                                              args.path_save_model, 
                                                              DEVICE) 
        
        for opt_item in ["optimizer_dis", "optimizer_gen"]:
            for state in optimizers[opt_item].state.values():
                for st1, st2 in state.items():
                    if isinstance(st2, torch.Tensor):
                        state[st1] = st2.to(DEVICE)              
        
        for sch_item in ["scheduler_dis", "scheduler_gen"]:
            for state in schedulers[sch_item].__dict__.values():
                if isinstance(state, torch.Tensor):
                    state.data = state.data.to(DEVICE)
                    if state._grad is not None:
                        state._grad.data = state._grad.data.to(DEVICE)                
    else:  
        model["dis_x"].apply(weights_init)
        model["dis_y"].apply(weights_init)
        model["gen_x"].apply(weights_init)
        model["gen_y"].apply(weights_init)     
                    
    model["dis_x"].train()
    model["dis_y"].train()
    model["gen_x"].train()
    model["gen_y"].train()            
    
    for epoch in range(args.epochs - load_epoch):
        print('*' * 50)
        print("Starting epoch [{}/{}]:".format(epoch+load_epoch+1, args.epochs))
        print("Learning rate for discriminator's optimizer = {}.".format(
            optimizers["optimizer_dis"].param_groups[0]["lr"]))
        print("Learning rate for generator's optimizer = {}.".format(
            optimizers["optimizer_gen"].param_groups[0]["lr"]))
        
        saved_files = glob.glob(args.path_save_images + "*")
        for file in saved_files:
            os.remove(file)

        for i, (real_x, real_y) in enumerate(tqdm(data_loader, leave=True)):

            real_x, real_y = real_x.to(DEVICE), real_y.to(DEVICE)

            with torch.cuda.amp.autocast() if is_amp else contextlib.ExitStack():

              fake_x = model["gen_y"](real_y)
              fake_y = model["gen_x"](real_x)
              reconstructed_x = model["gen_y"](fake_y)
              reconstructed_y = model["gen_x"](fake_x)            
            
            optimizers["optimizer_gen"].zero_grad()

            with torch.cuda.amp.autocast() if is_amp else contextlib.ExitStack():

              loss_id_x = criterions["crt_id"](model["gen_y"](real_x), real_x)
              loss_id_x *= args.lambda_xy * args.lambda_id
              loss_id_y = criterions["crt_id"](model["gen_x"](real_y), real_y)
              loss_id_y *= args.lambda_xy * args.lambda_id

              loss_gan_x = criterions["crt_gan"](model["dis_x"](fake_x), True)
              loss_gan_y = criterions["crt_gan"](model["dis_y"](fake_y), True)

              loss_cycle_x =  criterions["crt_cycle"](reconstructed_x, real_x)
              loss_cycle_x *= args.lambda_xy
              loss_cycle_y =  criterions["crt_cycle"](reconstructed_y, real_y)
              loss_cycle_y *= args.lambda_xy

              loss_gen = loss_gan_x + loss_gan_y + loss_cycle_x + loss_cycle_y
              loss_gen += loss_id_x + loss_id_y

            loss_gen.backward() 
            optimizers["optimizer_gen"].step()

            optimizers["optimizer_dis"].zero_grad()

            with torch.cuda.amp.autocast() if is_amp else contextlib.ExitStack():
              fake_y_tmp = fake_y_buffer.update(fake_y)
              pred_real_y = model["dis_y"](real_y)
              pred_fake_y = model["dis_y"](fake_y_tmp.detach())
              loss_dis_real_y = criterions["crt_gan"](pred_real_y, True)
              loss_dis_fake_y = criterions["crt_gan"](pred_fake_y, False)
              loss_dis_y = (loss_dis_real_y + loss_dis_fake_y) * 0.5

            loss_dis_y.backward()

            with torch.cuda.amp.autocast() if is_amp else contextlib.ExitStack():
              fake_x_tmp = fake_x_buffer.update(fake_x)
              pred_real_x = model["dis_x"](real_x)
              pred_fake_x = model["dis_x"](fake_x_tmp.detach())
              loss_dis_real_x = criterions["crt_gan"](pred_real_x, True)
              loss_dis_fake_x = criterions["crt_gan"](pred_fake_x, False)
              loss_dis_x = (loss_dis_real_x + loss_dis_fake_x) * 0.5

            loss_dis_x.backward()

            optimizers["optimizer_dis"].step()
       #--------------------------------
            if args.batch_size == 1 and (i % args.save_images_interval) == 0:
              save_image(torch.cat((
                  torch.cat((denorm(real_x), denorm(fake_y)), 3), 
                  torch.cat((denorm(real_y), denorm(fake_x)), 3)),2), 
                  os.path.join(args.path_save_images, 
                               f"XtoYandYtoX_epoch_{epoch+load_epoch+1}_itr_{i}.png"))
        #--------------------------------
        schedulers["scheduler_dis"].step()
        schedulers["scheduler_gen"].step()
        print('*' * 50)
        if (epoch+load_epoch+1) % args.save_model_interval == 0:
            save_checkpoint_state(epoch+load_epoch+1, model, optimizers, 
                                  schedulers, args.path_save_model)
                                
                                  
if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = arguments_initialization()
    print(args)
      
    isP100orCPU = True if DEVICE == 'cpu' else ifP100() #return true if cpu or P100 gpu
    is_amp = (1 - isP100orCPU) * args.use_amp # true if amp is activated       

    for dir in [args.path_save_images[:-1], args.path_save_model[:-1]]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    transform = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    tt.Normalize(*stats)])
    
    dataset = CycleGanDataset(args.path_to_data,  
                              transform, 
                              data_mode='train')

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        shuffle=args.use_shuffle, 
        num_workers=args.num_workers, 
        pin_memory=True
    ) 

    model = {
        "dis_x": Discriminator().to(DEVICE),
        "dis_y": Discriminator().to(DEVICE),
        "gen_x": Generator().to(DEVICE),
        "gen_y": Generator().to(DEVICE)
    }

    criterions = {
        "crt_gan": MSELossRedefined().to(DEVICE),
        "crt_cycle": nn.L1Loss(),
        "crt_id": nn.L1Loss()  
    }

    fit(model, criterions, data_loader, is_amp)
    