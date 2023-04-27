import numpy as np, os, time, gc, random
import argparse, torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
from torch.optim import lr_scheduler

from stage1_VAE.modules import resnet3D
from stage1_VAE.modules import patch_disc
from stage1_VAE.modules import decoder as net
from stage1_VAE.modules.loss_copy import Backward
from utils import auxiliaries as aux
from metrics.PyTorch_FVD import FVD_logging
from metrics.DTFVD import DTFVD_Score
from data.get_dataloder import get_loader

"""=========================Trainer Function==================================================="""


def trainer(opt, network, enc, disc_t, disc_s, disc_to, logger, epoch, data_loader, optimizer, backward):
    _ = network.train()

    logger.reset()
    data_iter = tqdm(data_loader, ascii=True, position=2)
    inp_string = 'Epoch {} || percep: ||coup_s: | coup_t:'.format(epoch)
    data_iter.set_description(inp_string)
    sequences = []

    for batch_idx, file_dict in enumerate(data_iter):

        ## Load data
        seq = file_dict["seq"].type(torch.FloatTensor).cuda()

        ## Compute Loss
        # sequnce : [seq_gen.detach().cpu(), seq_orig.cpu()]
        sequences = backward(network, enc, disc_t, disc_s, disc_to, seq, optimizer, epoch, logger)

        ## plot losses in console
        if batch_idx % opt.Training['verbose_idx'] == 0 and batch_idx or batch_idx == 5:
            loss_log = logger.get_iteration_mean(10)
            inp_string = 'Epoch {} || percep: {} || coup_s: {} | coup_t: {}'.format(
                epoch, np.round(loss_log[2], 2), np.round(loss_log[6], 2), np.round(loss_log[7], 2))
            data_iter.set_description(inp_string)


    ### Save images offline and in wandb
    gif = aux.plot_vid(opt, sequences, epoch, mode='train')
    wandb.log({"train_video": wandb.Video(gif, fps=3,  format="gif")})

    ### Empty GPU cache
    del sequences
    torch.cuda.empty_cache()

"""===========================Validation Function==================================================="""


def validator(opt, network, enc, logger, epoch, data_loader, backward):
    _ = network.eval()

    logger.reset()
    data_iter = tqdm(data_loader, position=2, ascii=True)
    inp_string = 'Epoch {} | Eval | L1: --- | Percep: ---'.format(epoch)
    data_iter.set_description(inp_string)

    with torch.no_grad():
        for image_idx, file_dict in enumerate(data_iter):

            # Load data, Sequence shape is BsxTxCxWxH tensor
            seq = file_dict["seq"].type(torch.FloatTensor).cuda()

            ## Compute Loss
            sequences = backward.eval(network, enc, seq, logger)

            if image_idx % opt.Training['verbose_idx'] == 0 and image_idx or image_idx == 5:
                loss_log = logger.get_iteration_mean(10)
                inp_string = 'Epoch {} | Eval | L1: {} | Percep: {}'.format(epoch, np.round(loss_log[0], 3),
                                                                            np.round(loss_log[1], 3))
                data_iter.set_description(inp_string)

    # Save images
    gif = aux.plot_vid(opt, sequences, epoch, mode='eval')
    wandb.log({"eval_video": wandb.Video(gif, fps=3,  format="gif")})

    # Empty GPU cache
    del sequences
    torch.cuda.empty_cache()

from metrics.PyTorch_FVD.FVD_logging import calculate_FVD as calc_FVD
from metrics.DTFVD.DTFVD_Score import calculate_FVD as calc_DTFVD

def evaluate_FVD_posterior(dloader, model, encoder, mode):
    _ = model.eval()
    seq_g, seq_o = [], []
    data_iter = tqdm(dloader, position=2, ascii=True)
    inp_string = 'FVD | Eval'
    data_iter.set_description(inp_string)

    with torch.no_grad():
        for image_idx, file in enumerate(data_iter):
            seq = file["seq"].type(torch.FloatTensor).cuda()
            motion, *_ = encoder(seq[:, 1:].transpose(1, 2))
            seq_gen = model(seq[:, 0], motion)
            #seq_g.append(seq_gen.cpu())
            #seq_o.append(seq[:, 1:].cpu())
            seq_g.append(seq_gen)
            seq_o.append(seq[:, 1:])
            inp_string = 'FVD | Eval | index: {}'.format(image_idx)
            data_iter.set_description(inp_string)
            torch.cuda.empty_cache()


    seq_gen  = torch.cat(seq_g, dim=0)
    seq_orig = torch.cat(seq_o, dim=0)

    seq_gen1, seq_gen2, seq_gen3, seq_gen4  = torch.chunk(seq_gen, 4, dim=0)
    seq_orig1, seq_orig2, seq_orig3, seq_orig4 = torch.chunk(seq_orig, 4, dim=0)

    torch.cuda.empty_cache()

    inp_string = 'before calc'
    data_iter.set_description(inp_string)
    print(inp_string)
    #FVD = calc_FVD(None, seq_orig, seq_gen, batch_size=1)
    #return FVD
    FVD1 = calc_FVD(None, seq_orig1.cpu(), seq_gen1.cpu(), batch_size=1)
    FVD2 = calc_FVD(None, seq_orig2.cpu(), seq_gen2.cpu(), batch_size=1)
    FVD3 = calc_FVD(None, seq_orig3.cpu(), seq_gen3.cpu(), batch_size=1)
    FVD4 = calc_FVD(None, seq_orig4.cpu(), seq_gen4.cpu(), batch_size=1)
    inp_string = 'FVD1: {} | FVD2: {} | FVD3: {} | FVD4: {}'.format(FVD1, FVD2, FVD3, FVD4)
    data_iter.set_description(inp_string)
    print(inp_string)
    return sum([FVD1, FVD2, FVD3])/3

def main(opt):

    """================= Create Model, Optimizer and Scheduler =========================="""
    decoder = net.Generator(opt.Decoder).cuda()
    disc_s  = patch_disc.NLayerDiscriminator(opt.Discriminator_Patch).cuda()
    disc_t  = resnet3D.Discriminator(opt.Discriminator_Temporal).cuda()
    disc_to  = resnet3D.Discriminator(opt.Discriminator_Temporal).cuda()
    encoder = resnet3D.Encoder(opt.Encoder).cuda()
    #I3D     = FVD_logging.load_model().cuda() if opt.Training['FVD']=='FVD' else DTFVD_Score.load_model(16).cuda()

    backward = Backward(opt)
    optimizer_AE = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=opt.Training['lr'],
                                       betas=(0.5, 0.9), weight_decay=opt.Training['weight_decay'])
    optimizer_3Dnet = torch.optim.Adam(disc_t.parameters(), lr=opt.Training['lr'], betas=(0.5, 0.9),
                                       weight_decay=opt.Training['weight_decay'])
    optimizer_3Dnet_op = torch.optim.Adam(disc_to.parameters(), lr=opt.Training['lr'], betas=(0.5, 0.9),
                                       weight_decay=opt.Training['weight_decay'])
    optimizer_2Dnet = torch.optim.Adam(disc_s.parameters(), lr=opt.Training['lr'], betas=(0.5, 0.9),
                                       weight_decay=opt.Training['weight_decay'])

    optimizer = [optimizer_AE, optimizer_3Dnet, optimizer_2Dnet, optimizer_3Dnet_op]

    scheduler_AE = lr_scheduler.ExponentialLR(optimizer_AE, gamma=opt.Training['lr_gamma'])
    scheduler_3Dnet = lr_scheduler.ExponentialLR(optimizer_3Dnet, gamma=opt.Training['lr_gamma'])
    scheduler_2Dnet = lr_scheduler.ExponentialLR(optimizer_2Dnet, gamma=opt.Training['lr_gamma'])
    scheduler_3Dnet_op = lr_scheduler.ExponentialLR(optimizer_3Dnet_op, gamma=opt.Training['lr_gamma'])
    

    """====================Reload model if needed ========================"""
    if opt.Training['reload_path']:

        pretrained_gen = torch.load(opt.Training['reload_path'] + '/latest_checkpoint_GEN.pth')
        _ = decoder.load_state_dict(pretrained_gen['state_dict'])

        pretrained_disc_t = torch.load(opt.Training['reload_path'] + '/latest_checkpoint_DISC_t.pth')
        _ = disc_t.load_state_dict(pretrained_disc_t['state_dict'])

        pretrained_disc_s = torch.load(opt.Training['reload_path'] + '/latest_checkpoint_DISC_s.pth')
        _ = disc_s.load_state_dict(pretrained_disc_s['state_dict'])

        pretrained_disc_op = torch.load(opt.Training['reload_path'] + '/latest_checkpoint_DISC_to.pth')
        _ = disc_to.load_state_dict(pretrained_disc_op['state_dict'])

        pretrained_enc = torch.load(opt.Training['reload_path'] + '/latest_checkpoint_ENC.pth')
        _ = encoder.load_state_dict(pretrained_enc['state_dict'])

        start_epoch = pretrained_gen['epoch']

        optimizer_AE.load_state_dict(pretrained_gen['optim_state_dict'])
        optimizer_3Dnet.load_state_dict(pretrained_disc_t['optim_state_dict'])
        optimizer_2Dnet.load_state_dict(pretrained_disc_s['optim_state_dict'])
        optimizer_3Dnet_op.load_state_dict(pretrained_disc_op['optim_state_dict'])

        scheduler_AE.load_state_dict(pretrained_gen['scheduler_state_dict'])
        scheduler_3Dnet.load_state_dict(pretrained_disc_t['scheduler_state_dict'])
        scheduler_3Dnet.load_state_dict(pretrained_disc_s['scheduler_state_dict'])
        scheduler_3Dnet_op.load_state_dict(pretrained_disc_op['scheduler_state_dict'])

        for param_group in optimizer_AE.param_groups:
            LR = param_group['lr']

        checkpoint_info = "Load checkpoint from '%s/latest_checkpoint.pth.tar' with LR %.8f" % (
            opt.Training['reload_path'], LR)
    else:
        start_epoch = 0
        checkpoint_info = "Starting from scratch with LR %.8f" % (opt.Training['lr'])

    print(checkpoint_info)

    """==================== Dataloader ========================"""
    dataset       = get_loader(opt.Data['dataset'])
    train_dataset = dataset.Dataset(opt, mode='train')
    eval_dataset  = dataset.Dataset(opt, mode='eval')

    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=True, drop_last=True)
    eval_data_loader  = torch.utils.data.DataLoader(eval_dataset, num_workers=opt.Training['workers'], drop_last=True,
                                                    batch_size=opt.Training['bs_eval'], shuffle=True)
    print("Batchsize for training: % 2d and for testing: % 2d" % (opt.Training['bs'], opt.Training['bs_eval']))

    """======================Set Logging Files======================"""
    dt = datetime.now()
    dt = '{}-{}-{}-{}-{}-{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    run_name = 'Stage1_' + opt.Data['dataset'] + '_Date-' + dt + '_' + opt.Training['savename']

    save_path = opt.Training['save_path'] + "/" + run_name

    ## Make the saving directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    opt.Training['save_path'] = save_path

    ## Create wandb logger for online logging
    log_dic = opt.Logging
    wandb.init(entity=log_dic['entitiy'], config=opt, dir=save_path, project=log_dic['project'],
               name=opt.Training['name'], mode=log_dic['mode'])
    # wandb.watch(model, log="all")

    ## Make folder to save video samples during training
    if not os.path.exists(save_path + '/videos'):
        os.makedirs(save_path + '/videos')

    ## save hyperparameters by saving yaml config
    OmegaConf.save(config=opt, f=save_path + "/config_stage1.yaml")

    ## Offline logging
    logging_keys_train = ["Loss_VAE", "Loss_L1", "LPIPS", "Loss_KL", "Loss_GEN_S", "Loss_GEN_T",
                          "Loss_Disc_T", "Loss_Fmap_T", "L_GP", "Logits_Real_T", "Logits_Fake_T", "Loss_Disc_S",
                          "Logits_Real_S", "Logits_Fake_S", "PSNR", "SSIM"]

    logging_keys_test = ["Loss_L1", "LPIPS", "L_KL", "PSNR", "SSIM", 'PFVD']

    log_train = aux.Logging(logging_keys_train)
    log_test  = aux.Logging(logging_keys_test[:-1])

    ### Setting up CSV writers
    full_log_train = aux.CSVlogger(save_path + "/log_per_epoch_train.csv", ["Epoch", "Time", "LR"] + logging_keys_train)
    full_log_test  = aux.CSVlogger(save_path + "/log_per_epoch_eval.csv", ["Epoch", "Time", "LR"] + logging_keys_test)

    """=================== Start training ! ==========================="""
    epoch_iterator = tqdm(range(start_epoch, opt.Training['n_epochs']), ascii=True, position=1)
    best_PFVD = 999

    torch.cuda.empty_cache()
    for epoch in epoch_iterator:
        epoch_time = time.time()
        lr = [group['lr'] for group in optimizer_2Dnet.param_groups][0]

        ## Training
        epoch_iterator.set_description("Training with lr={}".format(lr))
        trainer(opt, decoder, encoder, disc_t, disc_s, disc_to, log_train, epoch, train_data_loader, optimizer, backward)

        ## Validation
        epoch_iterator.set_description('Validating...')
        validator(opt, decoder, encoder, log_test, epoch, eval_data_loader, backward)
        ## Evaluate DTFVD
        epoch_iterator.set_description('Validating (DT)FVD score ...')
        #PFVD = evaluate_FVD_posterior(eval_data_loader, decoder, encoder, opt.Training['FVD'])
        #wandb.log({'FVD': PFVD})
        #PFVD = evaluate_FVD_posterior(eval_data_loader, decoder, encoder, opt.Training['FVD'])
        PFVD = aux.evaluate_FVD_posterior(eval_data_loader, decoder, encoder, None, opt.Training['FVD'])
        wandb.log({'FVD': PFVD})

        save_dict_GEN = aux.get_save_dict(decoder, optimizer_AE, scheduler_AE, epoch)
        save_dict_ENC = aux.get_save_dict(encoder, optimizer_AE, scheduler_AE, epoch)
        save_dict_DISC_t = aux.get_save_dict(disc_t, optimizer_3Dnet, scheduler_3Dnet, epoch)
        save_dict_DISC_s = aux.get_save_dict(disc_s, optimizer_2Dnet, scheduler_2Dnet, epoch)
        save_dict_DISC_to = aux.get_save_dict(disc_to, optimizer_3Dnet_op, scheduler_3Dnet_op, epoch)

        ## Save checkpoints to reload in case of process crashes
        torch.save(save_dict_GEN, save_path + '/latest_checkpoint_GEN.pth')
        torch.save(save_dict_ENC, save_path + '/latest_checkpoint_ENC.pth')
        torch.save(save_dict_DISC_t, save_path + '/latest_checkpoint_DISC_t.pth')
        torch.save(save_dict_DISC_s, save_path + '/latest_checkpoint_DISC_s.pth')
        torch.save(save_dict_DISC_to, save_path + '/latest_checkpoint_DISC_to.pth')

        ## SAVE CHECKPOINTS
        if PFVD < best_PFVD:
            torch.save(save_dict_GEN, save_path + '/best_PFVD_GEN.pth')
            torch.save(save_dict_ENC, save_path + '/best_PFVD_ENC.pth')
            best_PFVD = PFVD

        ## Perform scheduler update
        scheduler_AE.step()
        if epoch >= opt.Training['pretrain']:
            scheduler_3Dnet_op.step()
            scheduler_3Dnet.step()
            scheduler_2Dnet.step()

        ## Logg data from current epoch offline
        full_log_train.write([epoch, time.time() - epoch_time, lr, *log_train.log()])
        full_log_test.write([epoch, time.time() - epoch_time, lr, *log_test.log(), PFVD])


"""============================================"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", type=str, default='stage1_VAE/configs/bair_config.yaml', help="Define config file")
    parser.add_argument("-gpu", type=str, required=True)
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    """import tensorflow as tf
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices("GPU")[0], True)
    """
    
    aux.set_seed(42)
    main(conf)
