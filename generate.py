import torch
import os
import sys
import torchvision
import argparse
import torchvision.transforms as tt

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data.dataset import CycleGanDataset
from model.discriminator import Discriminator
from model.generator import Generator
from utils.denormalization import denorm

def arguments_initialization():
    parser = argparse.ArgumentParser(description="GAN project. Predicting images from test dataset")

    parser.add_argument('--path_to_data', type=str, default="datasets/current_dataset/", help="Path to data (data for each class should be separated to two test subfolders, e.g. (testX, testY) or (testA, testB)).")
    parser.add_argument('--path_load_model', type=str, default="checkpoints/", help="Path to load checkpoints.")
    parser.add_argument('--path_test_results', type=str, default="results_on_test/", help="Path to save results generated from test dataset.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of images (image's height is equal to width). Should be the same as for training procedure.")
    parser.add_argument("--load_epoch", type=int, required=True, help="Epoch to load pretrained model (note, that checkpoint for [--load_epoch] epoch should be in location [--path_load_model]).")
    parser.add_argument('--switch', action='store_true', default=True, help="Depend on this key X-->Y/Y-->X or identity X-->X/Y-->Y images would be generated")
    return parser.parse_args() 


def test_processing(path_to_test, path_load_model, transform, load_epoch, path_save):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = CycleGanDataset(path_to_test, transform=transform, data_mode='test')

    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False
        ) 

    model = {
        "dis_x": Discriminator().to(DEVICE),
        "dis_y": Discriminator().to(DEVICE),
        "gen_x": Generator().to(DEVICE),
        "gen_y": Generator().to(DEVICE)
    }

    path_to_load = os.path.join(path_load_model, 
    "checkpoint_epoch_{}.pth.tar".format(load_epoch))

    if os.path.isfile(path_to_load):  
        print("loading model from checkpoint for {} epoch".format(load_epoch))        
        checkpoint = torch.load(path_to_load, map_location=DEVICE)
        model["dis_x"].load_state_dict(checkpoint["st_dis_x"])
        model["dis_y"].load_state_dict(checkpoint["st_dis_y"])
        model["gen_x"].load_state_dict(checkpoint["st_gen_x"])
        model["gen_y"].load_state_dict(checkpoint["st_gen_y"])
    else:
          sys.exit("no checkpoint found for {} epoch".format(load_epoch))

    model["dis_x"].eval()
    model["dis_y"].eval()
    model["gen_x"].eval()
    model["gen_y"].eval()  

    for i, (real_x, real_y) in enumerate(tqdm(test_loader, leave=True)):
        real_x, real_y = real_x.to(DEVICE), real_y.to(DEVICE)
        with torch.no_grad():
            if args.switch:
                fake_x = model["gen_x"](real_y)
                fake_y = model["gen_y"](real_x)
            else:   
                fake_x = model["gen_y"](real_y)
                fake_y = model["gen_x"](real_x) 
        save_image(torch.cat((
            torch.cat((denorm(real_x), denorm(fake_y)), 3), 
            torch.cat((denorm(real_y), denorm(fake_x)), 3)),2), 
            os.path.join(path_save, 
            f"XtoYandYtoX_epoch_itr_{i}.png")) 


if __name__ == '__main__':
    args = arguments_initialization()
    print(args)

    for dir in [args.path_load_model[:-1], args.path_test_results[:-1]]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    transform = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size), 
                                    transforms.ToTensor(),
                                    tt.Normalize(*stats)])

    test_processing(args.path_to_data, args.path_load_model, transform, 
    args.load_epoch, args.path_test_results)