import torch
import sys
import os


def  save_checkpoint_state(epoch, model, optimizers, schedulers, path_save):

    checkpoint = {        
            "epoch": epoch,
            "st_dis_x": model["dis_x"].state_dict(),
            "st_dis_y": model["dis_y"].state_dict(),
            "st_gen_x": model["gen_x"].state_dict(),
            "st_gen_y": model["gen_y"].state_dict(), 
            "st_optimizer_dis": optimizers["optimizer_dis"].state_dict(),           
            "st_optimizer_gen": optimizers["optimizer_gen"].state_dict(),
            "st_scheduler_dis": schedulers["scheduler_dis"].state_dict(),
            "st_scheduler_gen": schedulers["scheduler_gen"].state_dict()
                }   
    path_to_save = os.path.join(path_save, f"checkpoint_epoch_{epoch}.pth.tar")            
    torch.save(checkpoint, path_to_save)


def  load_checkpoint_state(load_epoch, model, optimizers, schedulers, path_save, DEVICE):

    path_to_load = os.path.join(path_save, f"checkpoint_epoch_{load_epoch}.pth.tar")
    if os.path.isfile(path_to_load):  
        print(f"loading checkpoint for {load_epoch} epoch")        
        checkpoint = torch.load(path_to_load, map_location=DEVICE)
        model["dis_x"].load_state_dict(checkpoint["st_dis_x"])
        model["dis_y"].load_state_dict(checkpoint["st_dis_y"])
        model["gen_x"].load_state_dict(checkpoint["st_gen_x"])
        model["gen_y"].load_state_dict(checkpoint["st_gen_y"])
        optimizers["optimizer_dis"].load_state_dict(checkpoint["st_optimizer_dis"])
        optimizers["optimizer_gen"].load_state_dict(checkpoint["st_optimizer_gen"])
        schedulers["scheduler_dis"].load_state_dict(checkpoint["st_scheduler_dis"])
        schedulers["scheduler_gen"].load_state_dict(checkpoint["st_scheduler_gen"])
    else:
        #print(f"no checkpoint found for {load_epoch} epoch")
         sys.exit(f"no checkpoint found for {load_epoch} epoch")
    return model, optimizers, schedulers
