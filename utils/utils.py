import os
import numpy as np
import matplotlib.pyplot as plt

def rprint(str, rank):
    if rank==0:
        print(str)

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False

def imshow(img, norm=False):
    img = img.cpu().numpy()
    plt.imshow(np.transpose(np.array(img / 255 if norm else img, dtype=np.float32), (1, 2, 0)))
    plt.show()

def pl(a):
    plt.plot(a.cpu())
    plt.show()

def sc(a):
    plt.scatter(range(len(a.cpu())), a.cpu(), s=2, color='darkred', alpha=0.5)
    plt.show()

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def print_configuration(args, rank):
    dict = vars(args)
    if rank == 0:
        print('------------------Configurations------------------')
        for key in dict.keys():
            print("{}: {}".format(key, dict[key]))
        print('-------------------------------------------------')


# attack loader
from attack.fastattack import attack_loader
from tqdm import tqdm
def test_robustness(net, testloader, criterion, attack_list, rank):
    net.eval()
    test_loss = 0

    attack_module = {}
    for attack_name in attack_list:
        attack_module[attack_name] = attack_loader(net=net, attack=attack_name, eps=0.03, steps=30) \
                                                                                if attack_name != 'plain' else None

    for key in attack_module:
        total = 0
        correct = 0
        prog_bar = tqdm(enumerate(testloader), total=len(testloader), leave=False)
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            if key != 'plain':
                inputs = attack_module[key](inputs, targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[Test/%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (key, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

            # fast eval
            if (key == 'auto') or (key == 'fab'):
                if batch_idx >= int(len(testloader) * 0.3):
                    break

        rprint(f'{key}: {100. * correct / total:.2f}%', rank)