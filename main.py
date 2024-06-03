import warnings
warnings.filterwarnings("ignore")
import numpy as np
import shutil
import torch
import argparse
from utils_HSI import sample_gt, metrics, seed_worker, set_requires_grad
from datasets import get_dataset, HyperX
import time
import os
from datetime import datetime
from model.generator import SSDGnet
from model.Discriminator import discriminator
from model.MINet import ResnetGenerator
from losses import MI_loss

parser = argparse.ArgumentParser(description='PyTorch S2AMSnet')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/Houston/')
parser.add_argument('--source_name', type=str, default='Houston13',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Houston18',
                    help='the name of the test dir')

group_train = parser.add_argument_group('Training')
group_train.add_argument('--temp', type=float, default=0.07, help='temperature for contrastive loss function')
group_train.add_argument('--patch_size', type=int, default=13,
                    help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)Houston:11;Pavia:7")
group_train.add_argument('--lr', type=float, default=1e-3,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--batch_size', type=int, default=256,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--max_epoch', type=int, default=400)
group_train.add_argument('--test_stride', type=int, default=1,
                    help="Sliding window step stride during inference (default = 1)")
group_train.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
group_train.add_argument('--re_ratio', type=int, default=5,
                    help='multiple of of data augmentation')
group_train.add_argument('--seed', type=int, default=333,
                    help='random seed ')
group_train.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
group_train.add_argument('--log_interval', type=int, default=40)


group_model = parser.add_argument_group('model')
group_model.add_argument('--pro_dim', type=int, default=128)
group_model.add_argument("--GIN", type=bool, default=True, help='global intensity non-linear augmentation')
group_model.add_argument("--adv", type=bool, default=True, help='global intensity non-linear augmentation')
group_model.add_argument("--noise", type=bool, default=True, help='noise z')
group_model.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
group_model.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
group_model.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
group_model.add_argument('--GIN_ch', type=int, default=24, help='channel of GIN')


group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',default=False,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")
args = parser.parse_args()

def evaluate(net, val_loader, gpu):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(gpu)
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    results = metrics(ps, ys, n_classes=ys.max() + 1)
    return acc, results

def experiment():
    train_res = {
        'best_epoch': 0,
        'best_acc': 0,
        'Confusion_matrix': [],
        'OA': 0,
        'TPR': 0,
        'F1scores': 0,
        'kappa': 0,
        'finished': False
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hyperparams = vars(args)
    print(hyperparams)

    s = ''
    for k, v in args.__dict__.items():
        s += '\t' + k + '\t' + str(v) + '\n'

    f = open(log_dir + '/settings.txt', 'w+')
    f.write(s)
    f.close()

    seed_worker(args.seed) 
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                            args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                            args.data_path)

    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    tmp = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size']/2)+1
    img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
    gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))     

    train_gt_src, _, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    
    if tmp < 1:
        for i in range(args.re_ratio-1):
            img_src_con = np.concatenate((img_src_con,img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
           

    hyperparams_train = hyperparams.copy()
    hyperparams_train['flip_augmentation'] = True
    hyperparams_train['radiation_augmentation'] = True

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=True)

    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                    pin_memory=True,
                                    batch_size=hyperparams['batch_size'])           

    D_net = discriminator(inchannel=N_BANDS, outchannel=args.pro_dim, num_classes=num_classes, patch_size=hyperparams['patch_size']).to(args.gpu)
    D_opt = torch.optim.Adam(D_net.parameters(), lr=args.lr)
    G_net = SSDGnet(args).to(args.gpu)
    G_opt = torch.optim.Adam(G_net.parameters(), lr=args.lr)
    MINet = ResnetGenerator(input_nc=N_BANDS, output_nc=N_BANDS, ngf=8, norm_layer=torch.nn.InstanceNorm2d,
                          use_dropout=False, no_antialias=False, no_antialias_up=False, n_blocks=6).to(args.gpu)
    flag_MI = True
    cls_criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    taracc, taracc_list = 0, []
    for epoch in range(1,args.max_epoch+1):
        t1 = time.time()    
        loss_list = []
        D_net.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1
            D_opt.zero_grad()

            G_opt.zero_grad()

            aug_img1, aug_img2 = G_net(x)
            alpha1 = np.random.beta(0.6, 0.6)
            alpha2 = np.random.beta(0.6, 0.6)
            mix_img1 = alpha1 * aug_img1 + (1 - alpha1) * x
            mix_img2 = alpha2 * aug_img2 + (1 - alpha2) * x

           
            loss_MI = MI_loss(x, aug_img2, MINet, args) + MI_loss(x, aug_img1, MINet, args)

            if flag_MI:
                opt_MI = torch.optim.Adam(MINet.parameters(), lr=args.lr)
                opt_MI.zero_grad()
                flag_MI = False
                loss_MI = MI_loss(x, aug_img2, MINet, args) + MI_loss(x, aug_img1, MINet, args)

            predict1 = D_net(aug_img1.detach())
            predict2 = D_net(aug_img2.detach())
            predict3 = D_net(mix_img1.detach())
            predict4 = D_net(mix_img2.detach())
            
            loss_aug1 = cls_criterion(predict1, y.long())
            loss_aug2 = cls_criterion(predict2, y.long())
            loss_aug3 = cls_criterion(predict3, y.long())
            loss_aug4 = cls_criterion(predict4, y.long())

            prob1 = torch.softmax(predict1, dim=1)
            prob2 = torch.softmax(predict2, dim=1)
            prob3 = torch.softmax(predict3, dim=1)
            prob4 = torch.softmax(predict4, dim=1)

            loss_kl = torch.nn.KLDivLoss()(prob1, prob4) + torch.nn.KLDivLoss()(prob2, prob3)

            loss_min = loss_kl + loss_aug1 + loss_aug2+ loss_aug3 + loss_aug4
            loss_min.backward()
            D_opt.step()

            set_requires_grad(D_net, False)
            predict1 = D_net(aug_img1)
            predict2 = D_net(aug_img2)
            predict3 = D_net(mix_img1)
            predict4 = D_net(mix_img2)

            prob1 = torch.softmax(predict1, dim=1)
            prob2 = torch.softmax(predict2, dim=1)
            prob3 = torch.softmax(predict3, dim=1)
            prob4 = torch.softmax(predict4, dim=1)

            loss_kl = torch.nn.KLDivLoss()(prob1, prob4) + torch.nn.KLDivLoss()(prob2, prob3)

            loss_max = -loss_kl + loss_MI 
            loss_max.backward()
            set_requires_grad(D_net, True)
            D_opt.step()
            G_opt.step()
            opt_MI.step()
           
            loss_list.append([loss_max.item(), loss_min.item()])
        loss_max, loss_min = np.mean(loss_list, 0)
        
        t2 = time.time()
        D_net.eval()
        taracc, results = evaluate(D_net, test_loader, args.gpu)
        if best_acc < taracc:
            best_acc = taracc
            torch.save({'Discriminator': D_net.state_dict()}, os.path.join(log_dir, f'best.pth'))
            train_res['best_epoch'] = epoch
            train_res['best_acc'] = '{:.2f}'.format(best_acc)
            train_res['Confusion_matrix'] = '{:}'.format(results['Confusion_matrix'])
            train_res['OA'] = '{:.2f}'.format(results['Accuracy'])
            train_res['TPR'] = '{:}'.format(np.round(results['TPR'] * 100, 2))
            train_res['F1scores'] = '{:}'.format(results["F1_scores"])
            train_res['kappa'] = '{:.4f}'.format(results["Kappa"])
        print(
            f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f}, loss_min {loss_min:.4f} loss_max {loss_max:.4f} /// Test {len(test_loader.dataset)}, taracc {taracc:.2f}')

    with open(log_dir + '/train_log.txt', 'w+') as f:
        for key, value in train_res.items():
            f.write(f"{key}: {value}\n")
    f.close()
    
if __name__=='__main__':
    repeat_time = 10
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%Y%m%d%H%M%S')
    exp_name = '{}/{}'.format(args.save_path, args.source_name+'to'+args.target_name+'_'+time_str)
    for i in range(repeat_time):
        timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_dir = os.path.join(BASE_DIR, exp_name, 'lr_' + str(args.lr) +
                           '_pt' + str(args.patch_size) + '_bs' + str(args.batch_size) + '_' +timestamp)
        log_dir = log_dir.replace('\\', '/')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        experiment()













