import pickle
import os
import shutil
import argparse
import itertools

import torch
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from pytorch_fid.fid_score import calculate_fid_given_paths

from datasets import ImageDataset
from utils import weights_init_normal, LambdaLR, ReplayBuffer
from models import Generator, Discriminator, Global_Discriminator

# FID Parameters
IDENTITY_LOSS_FID_SCALING_FACTOR = 200
IDENTITY_LOSS_FID_PROPORTION = 0.5
CYCLIC_CONSISTENCY_LOSS_FID_SCALING_FACTOR = 100
CYCLIC_CONSISTENCY_LOSS_FID_PROPORTION = 0.2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epoch', 
        type=int, 
        default=0, 
        help='starting epoch')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=120,
        help='number of epochs training')
    parser.add_argument(
        '--batchSize',
        type=int,
        default=3,
        help='size of the batch sizes')
    parser.add_argument(
        '--dataroot',
        type=str,
        required=True,
        help='root directory of the dataset')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0002,
        help='initial learning rate')
    parser.add_argument(
        '--decay_epoch',
        type=int,
        default=25,
        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='size of the data crop (squared assumed)')
    parser.add_argument(
        '--input_nc',
        type=int,
        default=3,
        help='number of channels of the input data')
    parser.add_argument(
        '--output_nc', 
        type=int, 
        default=3,
        help='number of channels of the output data')
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=1,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--n_gpu',
        type=int,
        default=0,
        help='number of GPUs to use for training')
    opt = parser.parse_args()
    print(opt)

    ### definition of Variables #####

    device = torch.device("cuda" if torch.cuda.is_available() and opt.n_gpu > 0 else "cpu")

    netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
    netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)
    netD_A = Discriminator(opt.input_nc).to(device)
    netD_B = Discriminator(opt.input_nc).to(device)
    netDg_A = Global_Discriminator().to(device)
    netDg_B = Global_Discriminator().to(device)

    # MultiGPU support
    if device == "cuda":
        gpu_count = min(opt.n_gpu, torch.cuda.device_count())
        if  gpu_count > 1:
            netG_A2B = torch.nn.DataParallel(netG_A2B, list(range(gpu_count)))
            netG_B2A = torch.nn.DataParallel(netG_B2A, list(range(gpu_count)))
            netD_A = torch.nn.DataParallel(netD_A, list(range(gpu_count)))
            netD_B = torch.nn.DataParallel(netD_B, list(range(gpu_count)))
            netDg_A = torch.nn.DataParallel(netDg_A, list(range(gpu_count)))
            netDg_B = torch.nn.DataParallel(netDg_B, list(range(gpu_count)))

    # Initialize Weights
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
    netDg_A.apply(weights_init_normal)
    netDg_B.apply(weights_init_normal)

    # Define Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    Tensor = torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size).to(device)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size).to(device)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0).to(device), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0).to(device), requires_grad=False)

    ## Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(
        itertools.chain(
            netG_A2B.parameters(),
            netG_B2A.parameters()),
        lr=opt.lr,
        betas=(
            0.5,
            0.999))
    optimizer_D_A = torch.optim.Adam(
        netD_A.parameters(),
        lr=opt.lr,
        betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(
        netD_B.parameters(),
        lr=opt.lr,
        betas=(0.5, 0.999))
    optimizer_netDg_A = torch.optim.Adam(
        netDg_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_netDg_B = torch.optim.Adam(
        netDg_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(
            opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(
            opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(
            opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_netDg_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_netDg_A, lr_lambda=LambdaLR(
            opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_netDg_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_netDg_B, lr_lambda=LambdaLR(
            opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs and Targets Memory Allocation
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    fake_Ag_buffer = ReplayBuffer()
    fake_Bg_buffer = ReplayBuffer()


    # Prep for FID
    fakes_dir = "./temp/fakes"
    reals_dir = "./temp/reals"

    if os.path.exists(reals_dir):
        shutil.rmtree(reals_dir)
    if os.path.exists(fakes_dir):
        shutil.rmtree(fakes_dir)
    os.makedirs(reals_dir)
    os.makedirs(fakes_dir)


    # Dataset Loader

    transforms_ = [
        transforms.Resize(int(opt.size * 1.12), transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(opt.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    #### A-> Synthetic Pizza ________ B -> Real Pizza ####
    dataloader = DataLoader(
        ImageDataset(
            opt.dataroot,
            transforms_=transforms_,
            unaligned=True),
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.n_cpu)

    ##############################################
    loss_dict = {}
    curr = 0
    ############### Training #####################
    torch.cuda.empty_cache()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))
            ##### Generators A2B and B2A #########
            optimizer_G.zero_grad()

            # Identity Loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # Identity FID
            # Penalize large FID between G_A2B(B) and B if real B is fed
            for j in range(opt.batchSize):
                save_image(real_B[j], os.path.join(reals_dir, "real_" + str(j) + ".png"))
                save_image(same_B[j], os.path.join(fakes_dir, "fake_" + str(j) + ".png"))
            fid_identity_B = float(calculate_fid_given_paths([reals_dir, fakes_dir], 1, device, dims=2048)) / IDENTITY_LOSS_FID_SCALING_FACTOR
            # Penalize large FID between G_B2A(B) and A if real A is fed
            for j in range(opt.batchSize):
                save_image(real_A[j], os.path.join(reals_dir, "real_" + str(j) + ".png"))
                save_image(same_A[j], os.path.join(fakes_dir, "fake_" + str(j) + ".png"))
                
            fid_identity_A = float(calculate_fid_given_paths([reals_dir, fakes_dir], 1, device, dims=2048)) / IDENTITY_LOSS_FID_SCALING_FACTOR

            #print(loss_identity_A, fid_identity_A, loss_identity_B, fid_identity_B, sum([loss_identity_A, fid_identity_A, loss_identity_B, fid_identity_B]))

            # Net Identity Loss
            net_identity_loss_A = loss_identity_A * (1 - IDENTITY_LOSS_FID_PROPORTION) + fid_identity_A * IDENTITY_LOSS_FID_PROPORTION
            net_identity_loss_B = loss_identity_B * (1 - IDENTITY_LOSS_FID_PROPORTION) + fid_identity_B * IDENTITY_LOSS_FID_PROPORTION

            ###################################################

            # GAN Adversarial loss
            # A to B
            fake_B = netG_A2B(real_A)
            # PatchGAN Discriminator
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(torch.squeeze(pred_fake, 1), target_real)
            # Global Discriminator
            pred_fake_g = netDg_B(fake_B)
            loss_GAN_A2B_g = criterion_GAN(torch.squeeze(pred_fake_g, 1), target_real)
            # B to A
            fake_A = netG_B2A(real_B)
            # PatchGAN Discriminator
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(torch.squeeze(pred_fake, 1), target_real)
            # Global Discriminator
            pred_fake_g = netDg_A(fake_A)
            loss_GAN_B2A_g = criterion_GAN(torch.squeeze(pred_fake_g, 1), target_real)

            # Net GAN Adverserial Loss
            net_gan_loss_A2B = loss_GAN_A2B + loss_GAN_A2B_g
            net_gan_loss_B2A = loss_GAN_B2A + loss_GAN_B2A_g

            ###################################################

            # Cyclic Consistency Loss
            reconstructed_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(reconstructed_A, real_A) * 10.0
            reconstructed_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(reconstructed_B, real_B) * 10.0

            # Cyclic Consistency FID
            # Penalize large FID between real A and reconstructed A
            for j in range(opt.batchSize):
                save_image(real_A[j], os.path.join(reals_dir, "real_" + str(j) + ".png"))
                save_image(reconstructed_A[j], os.path.join(fakes_dir, "fake_" + str(j) + ".png"))
            fid_ABA = float(calculate_fid_given_paths([reals_dir, fakes_dir], 1, device, dims=2048)) / CYCLIC_CONSISTENCY_LOSS_FID_SCALING_FACTOR
            # Penalize large FID between real B and reconstructed B
            for j in range(opt.batchSize):
                save_image(real_B[j], os.path.join(reals_dir, "real_" + str(j) + ".png"))
                save_image(reconstructed_B[j], os.path.join(fakes_dir, "fake_" + str(j) + ".png"))      
            fid_BAB = float(calculate_fid_given_paths([reals_dir, fakes_dir], 1, device, dims=2048)) / CYCLIC_CONSISTENCY_LOSS_FID_SCALING_FACTOR

            #print(loss_cycle_ABA, fid_ABA, loss_cycle_BAB, fid_BAB, sum([loss_cycle_ABA, fid_ABA, loss_cycle_BAB, fid_BAB]))

            # Net Cyclic Consistency Loss
            net_cycle_loss_ABA = loss_cycle_ABA * (1 - CYCLIC_CONSISTENCY_LOSS_FID_PROPORTION) + fid_ABA * CYCLIC_CONSISTENCY_LOSS_FID_PROPORTION
            net_cycle_loss_BAB = loss_cycle_BAB * (1 - CYCLIC_CONSISTENCY_LOSS_FID_PROPORTION) + fid_BAB * CYCLIC_CONSISTENCY_LOSS_FID_PROPORTION

            # Total loss
            loss_G = net_identity_loss_A + net_identity_loss_B + net_gan_loss_A2B + net_gan_loss_B2A + net_cycle_loss_ABA + net_cycle_loss_BAB
            loss_G.backward()

            optimizer_G.step()
        

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(torch.squeeze(pred_real, 1), target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(torch.squeeze(pred_fake, 1), target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(torch.squeeze(pred_real, 1), target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(torch.squeeze(pred_fake, 1), target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            ###### Global Discriminator A2B ######
            optimizer_netDg_A.zero_grad()

            # Real loss
            pred_real = netDg_A(real_A)
            loss_D_real = criterion_GAN(torch.squeeze(pred_real, 1), target_real)

            # Fake loss
            fake_Ag = fake_Ag_buffer.push_and_pop(fake_A)
            pred_fake = netDg_A(fake_Ag.detach())
            loss_D_fake = criterion_GAN(torch.squeeze(pred_fake, 1), target_fake)

            # Total loss
            loss_D_Ag = (loss_D_real + loss_D_fake) * 0.5
            loss_D_Ag.backward()

            optimizer_netDg_A.step()
            ###################################

            ###### Global Discriminator B2A ######
            optimizer_netDg_B.zero_grad()

            # Real loss
            pred_real = netDg_B(real_B)
            loss_D_real = criterion_GAN(torch.squeeze(pred_real, 1), target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netDg_B(fake_B.detach())
            loss_D_fake = criterion_GAN(torch.squeeze(pred_fake, 1), target_fake)

            # Total loss
            loss_D_Bg = (loss_D_real + loss_D_fake) * 0.5
            loss_D_Bg.backward()

            optimizer_netDg_B.step()
            ###################################
            
            print("Epoch: {}\tBatch: {} / {}".format(epoch + 1, i + 1, len(dataloader)))
            loss_G_identity = net_identity_loss_A + net_identity_loss_B
            loss_G_GAN = (loss_GAN_A2B + loss_GAN_B2A)
            loss_G_cycle = net_cycle_loss_ABA + net_cycle_loss_BAB
            loss_D = (loss_D_A + loss_D_B + loss_D_Ag + loss_D_Bg)
            print({'loss_G': loss_G.item(),
                'loss_G_identity': loss_G_identity.item(),
                'loss_G_GAN': loss_G_GAN.item(),
                'loss_G_cycle': loss_G_cycle.item(),
                'loss_D': loss_D.item()})
        
            loss_dict[curr] = {}
            loss_dict[curr]['loss_G'] = loss_G
            loss_dict[curr]['loss_G_identity'] = (loss_G_identity)
            loss_dict[curr]['loss_G_GAN'] = (loss_GAN_A2B + loss_GAN_B2A)
            loss_dict[curr]['loss_G_cycle'] = (loss_G_cycle)
            loss_dict[curr]['loss_D'] = (loss_D)
            curr += 1
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        lr_scheduler_netDg_A.step()
        lr_scheduler_netDg_B.step()
        idx = 0

        os.makedirs('output/' + str(epoch + 1), exist_ok=True)
        torch.save(netG_A2B.state_dict(), 'output/' + str(epoch + 1) + '/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/' + str(epoch + 1) + '/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/' + str(epoch + 1) + '/netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/' + str(epoch + 1) + '/netD_B.pth')
        torch.save(netDg_A.state_dict(), 'output/' + str(epoch + 1) + '/netDg_A.pth')
        torch.save(netDg_B.state_dict(), 'output/' + str(epoch + 1) + '/netDg_B.pth')
        with open('output/' + str(epoch + 1) + '/loss_dict.pickle', 'wb') as handle:
            pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i, batch in enumerate(dataloader):
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        fake_B = netG_A2B(real_A)
        fake_A = netG_B2A(real_B)
        save_image(real_A, 'output/' + str(epoch) + '/real_A.png')
        save_image(real_B, 'output/' + str(epoch) + '/real_B.png')
        save_image(fake_A, 'output/' + str(epoch) + '/fake_A.png')
        save_image(fake_B, 'output/' + str(epoch) + '/fake_B.png')
        idx += 1
        if idx == 1:
            break
