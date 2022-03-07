from utils import CMU_motion_3d as CMU_Motion3D
from model import stage_4
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')

    net_pred = stage_4.MultiStageModel(opt=opt)
    net_pred.to(opt.cuda_idx)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        else:
            model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.DatasetsSmooth(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = CMU_Motion3D.CMU_Motion3D(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = CMU_Motion3D.CMU_Motion3D(opt, split=2)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)
    test_loader = {}
    acts = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
            "washwindow"]
    for act in acts:
        test_dataset = CMU_Motion3D.CMU_Motion3D(opt=opt, split=2, actions=act)
        dim_used = dataset.dim_used
        test_loader[act] = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True)

    # test_dataset = CMU_Motion3D.CMU_Motion3D(opt, split=2)
    # print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    # test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
    #                          pin_memory=True)

    dim_used = dataset.dim_used

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, dim_used=dim_used)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))

            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt, dim_used=dim_used)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))

            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo, dim_used=dim_used)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))

            test_error = 0
            for act in acts:
                ret_test = run_model(net_pred, is_train=3, data_loader=test_loader[act], opt=opt, epo=epo, dim_used=dim_used)
                for j in range(1, 26):
                    test_error += ret_test["#{:d}ms".format(j * 40)]

            test_error = test_error / (25*len(acts))
            print('testing error: {:.3f}'.format(test_error))

            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            #
            # if ret_valid['m_p3d_h36'] < err_best:
            #     err_best = ret_valid['m_p3d_h36']
            #     is_best = True

            if test_error < err_best:
                err_best = test_error
                is_best = True

            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)

def eval(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    net_pred = stage_4.MultiStageModel(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    # load model
    model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len, map_location='cuda:0')
    net_pred.load_state_dict(ckpt['state_dict'])

    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
    acts = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
               "washwindow"]

    data_loader = {}

    for act in acts:
        dataset = CMU_Motion3D.CMU_Motion3D(opt=opt, split=2, actions=act)
        dim_used = dataset.dim_used
        data_loader[act] = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True)

    valid_dataset = CMU_Motion3D.CMU_Motion3D(opt, split=2)
    print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
    valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                              pin_memory=True)


    ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, dim_used=dim_used)
    print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
    # do test
    is_create = True
    avg_ret_log = []

    for act in acts:
        ret_test = run_model(net_pred, is_train=3, data_loader=data_loader[act], opt=opt, dim_used=dim_used)
        ret_log = np.array([act])
        head = np.array(['action'])

        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, ['test_' + k])

        avg_ret_log.append(ret_log[1:])
        log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
        is_create = False

    avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
    avg_ret_log = np.mean(avg_ret_log, axis=0)

    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)

def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i+1], dim=1)
    return smooth_data

def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, dim_used=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = (np.array(range(opt.output_n)) + 1)*40
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n

    seq_in = opt.kernel_size
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 1
    # idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
    #         out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    for i, (p3d_h36) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, all_dim = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().to(opt.cuda_idx)

        smooth1 = smooth(p3d_h36[:, :, dim_used],
                         sample_len=opt.kernel_size + opt.output_n,
                         kernel_size=opt.kernel_size).clone()

        smooth2 = smooth(smooth1,
                         sample_len=opt.kernel_size + opt.output_n,
                         kernel_size=opt.kernel_size).clone()

        smooth3 = smooth(smooth2,
                         sample_len=opt.kernel_size + opt.output_n,
                         kernel_size=opt.kernel_size).clone()

        input = p3d_h36[:, :, dim_used].clone()

        p3d_sup_4 = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_sup_3 = smooth1.clone()[:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_sup_2 = smooth2.clone()[:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_sup_1 = smooth3.clone()[:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])

        p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1 = net_pred(input, input_n=in_n, output_n=out_n, itera=itera)

        p3d_out_4 = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out_4[:, :, dim_used] = p3d_out_all_4[:, seq_in:]
        p3d_out_4[:, :, index_to_ignore] = p3d_out_4[:, :, index_to_equal]
        p3d_out_4 = p3d_out_4.reshape([-1, out_n, all_dim//3, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, all_dim//3, 3])

        p3d_out_all_4 = p3d_out_all_4.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_out_all_3 = p3d_out_all_3.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_out_all_2 = p3d_out_all_2.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_out_all_1 = p3d_out_all_1.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            loss_p3d_4 = torch.mean(torch.norm(p3d_out_all_4 - p3d_sup_4, dim=3))
            loss_p3d_3 = torch.mean(torch.norm(p3d_out_all_3 - p3d_sup_3, dim=3))
            loss_p3d_2 = torch.mean(torch.norm(p3d_out_all_2 - p3d_sup_2, dim=3))
            loss_p3d_1 = torch.mean(torch.norm(p3d_out_all_1 - p3d_sup_1, dim=3))

            loss_all = (loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1)/4
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d_4.cpu().data.numpy() * batch_size


        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out_4, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out_4, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}ms".format(titles[j])] = m_p3d_h36[j]
    return ret

if __name__ == '__main__':

    option = Options().parse()

    if option.is_eval == False:
        main(option)
    else:
        eval(option)