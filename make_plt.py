import numpy as np
import matplotlib.pyplot as plt
import sys
import os

filename = sys.argv[-1]
# filename = './log/log_sbm50_link_pred_egcn_h_20220911183855_r0.log'
file_log = filename.split('/')[-1][0:-4]
new_dir_path = './log/img_'+file_log
microavg_path = '/microavg'
class0_path = '/class0'
class1_path = '/class1'

try:
    os.mkdir(new_dir_path)
except FileExistsError:
    pass

try:
    os.mkdir(new_dir_path+microavg_path)
except FileExistsError:
    pass
try:
    os.mkdir(new_dir_path+class0_path)
except FileExistsError:
    pass
try:
    os.mkdir(new_dir_path+class1_path)
except FileExistsError:
    pass


TRAIN_errors = [[],[]]
TRAIN_losses = [[],[]]
VALID_errors = [[],[]]
VALID_losses = [[],[]]
TEST_errors = [[],[]]
TEST_losses = [[],[]]


TRAIN_MAP = [[],[]]
VALID_MAP = [[],[]]
TEST_MAP = [[],[]]


TRAIN_MRR = [[],[]]
VALID_MRR = [[],[]]
TEST_MRR = [[],[]]


TRAIN_microavg_pre = [[],[]]
VALID_microavg_pre = [[],[]]
TEST_microavg_pre = [[],[]]

TRAIN_microavg_rec = [[],[]]
VALID_microavg_rec = [[],[]]
TEST_microavg_rec = [[],[]]

TRAIN_microavg_f1 = [[],[]]
VALID_microavg_f1 = [[],[]]
TEST_microavg_f1 = [[],[]]

TRAIN_class0_pre = [[],[]]
VALID_class0_pre = [[],[]]
TEST_class0_pre = [[],[]]

TRAIN_class0_rec = [[],[]]
VALID_class0_rec = [[],[]]
TEST_class0_rec = [[],[]]

TRAIN_class0_f1 = [[],[]]
VALID_class0_f1 = [[],[]]
TEST_class0_f1 = [[],[]]

TRAIN_class1_pre = [[],[]]
VALID_class1_pre = [[],[]]
TEST_class1_pre = [[],[]]

TRAIN_class1_rec = [[],[]]
VALID_class1_rec = [[],[]]
TEST_class1_rec = [[],[]]

TRAIN_class1_f1 = [[],[]]
VALID_class1_f1 = [[],[]]
TEST_class1_f1 = [[],[]]

with open(filename) as f:
    for line in f:
        line=line.replace('INFO:root:','').replace('\n','')
        


        if 'TRAIN epoch' in line or 'VALID epoch' in line or 'TEST epoch' in line:
            set = line.split(' ')[1]
            epoch = int(line.split(' ')[3])

            
        elif 'mean errors' in line:
            v=float(line.split('mean errors ')[1])#float(line.split('(')[1].split(')')[0])
            if set == 'TRAIN':
                TRAIN_errors[0].append(epoch)
                TRAIN_errors[1].append(v)
            elif set == 'VALID':
                VALID_errors[0].append(epoch)
                VALID_errors[1].append(v)
            elif set == 'TEST':
                TEST_errors[0].append(epoch)
                TEST_errors[1].append(v)
            
        elif 'mean losses' in line:
            v = float(line.split('(')[1].split(')')[0].split(',')[0])
            if set == 'TRAIN':
                TRAIN_losses[0].append(epoch)
                TRAIN_losses[1].append(v)
            elif set == 'VALID':
                VALID_losses[0].append(epoch)
                VALID_losses[1].append(v)
            elif set == 'TEST':
                TEST_losses[0].append(epoch)
                TEST_losses[1].append(v)
        
        elif 'mean MRR' in line:
            mrr=float(line.split('mean MRR ')[1].split(' ')[0])
            map=float(line.split('mean MAP ')[1].split(' ')[0])

            if set == 'TRAIN':
                TRAIN_MRR[0].append(epoch)
                TRAIN_MRR[1].append(mrr)

                TRAIN_MAP[0].append(epoch)
                TRAIN_MAP[1].append(map)
            elif set == 'VALID':
                VALID_MRR[0].append(epoch)
                VALID_MRR[1].append(mrr)

                VALID_MAP[0].append(epoch)
                VALID_MAP[1].append(map)
            elif set == 'TEST':
                TEST_MRR[0].append(epoch)
                TEST_MRR[1].append(mrr)

                TEST_MAP[0].append(epoch)
                TEST_MAP[1].append(map)    


        elif 'measures microavg' in line:
            # v=float(line.split('mean MRR ')[1].split(' ')[0])
            pre = float(line.split('precision ')[1].split(' ')[0])
            rec = float(line.split('recall ')[1].split(' ')[0])
            f1 = float(line.split('f1 ')[1].split(' ')[0])
            if set == 'TRAIN':
                TRAIN_microavg_pre[0].append(epoch)
                TRAIN_microavg_pre[1].append(pre)

                TRAIN_microavg_rec[0].append(epoch)
                TRAIN_microavg_rec[1].append(rec)

                TRAIN_microavg_f1[0].append(epoch)
                TRAIN_microavg_f1[1].append(f1)

            elif set == 'VALID':
                VALID_microavg_pre[0].append(epoch)
                VALID_microavg_pre[1].append(pre)

                VALID_microavg_rec[0].append(epoch)
                VALID_microavg_rec[1].append(rec)

                VALID_microavg_f1[0].append(epoch)
                VALID_microavg_f1[1].append(f1)

            elif set == 'TEST':
                TEST_microavg_pre[0].append(epoch)
                TEST_microavg_pre[1].append(pre)

                TEST_microavg_rec[0].append(epoch)
                TEST_microavg_rec[1].append(rec)

                TEST_microavg_f1[0].append(epoch)
                TEST_microavg_f1[1].append(f1)

        elif 'measures for class 0' in line:
            # v=float(line.split('mean MRR ')[1].split(' ')[0])
            pre = float(line.split('precision ')[1].split(' ')[0])
            rec = float(line.split('recall ')[1].split(' ')[0])
            f1 = float(line.split('f1 ')[1].split(' ')[0])
            if set == 'TRAIN':
                TRAIN_class0_pre[0].append(epoch)
                TRAIN_class0_pre[1].append(pre)

                TRAIN_class0_rec[0].append(epoch)
                TRAIN_class0_rec[1].append(rec)

                TRAIN_class0_f1[0].append(epoch)
                TRAIN_class0_f1[1].append(f1)

            elif set == 'VALID':
                VALID_class0_pre[0].append(epoch)
                VALID_class0_pre[1].append(pre)

                VALID_class0_rec[0].append(epoch)
                VALID_class0_rec[1].append(rec)

                VALID_class0_f1[0].append(epoch)
                VALID_class0_f1[1].append(f1)
                
            elif set == 'TEST':
                TEST_class0_pre[0].append(epoch)
                TEST_class0_pre[1].append(pre)

                TEST_class0_rec[0].append(epoch)
                TEST_class0_rec[1].append(rec)

                TEST_class0_f1[0].append(epoch)
                TEST_class0_f1[1].append(f1)

        elif 'measures for class 1' in line:
            # v=float(line.split('mean MRR ')[1].split(' ')[0])
            pre = float(line.split('precision ')[1].split(' ')[0])
            rec = float(line.split('recall ')[1].split(' ')[0])
            f1 = float(line.split('f1 ')[1].split(' ')[0])
            if set == 'TRAIN':
                TRAIN_class1_pre[0].append(epoch)
                TRAIN_class1_pre[1].append(pre)

                TRAIN_class1_rec[0].append(epoch)
                TRAIN_class1_rec[1].append(rec)

                TRAIN_class1_f1[0].append(epoch)
                TRAIN_class1_f1[1].append(f1)

            elif set == 'VALID':
                VALID_class1_pre[0].append(epoch)
                VALID_class1_pre[1].append(pre)

                VALID_class1_rec[0].append(epoch)
                VALID_class1_rec[1].append(rec)

                VALID_class1_f1[0].append(epoch)
                VALID_class1_f1[1].append(f1)
                
            elif set == 'TEST':
                TEST_class1_pre[0].append(epoch)
                TEST_class1_pre[1].append(pre)

                TEST_class1_rec[0].append(epoch)
                TEST_class1_rec[1].append(rec)

                TEST_class1_f1[0].append(epoch)
                TEST_class1_f1[1].append(f1)



            if epoch==50000:
                break



def make_plot_data(TRAIN_list,VALID_list,TEST_list):
    # print(TRAIN_list[0])
    # print(TRAIN_list[1])
    # print(VALID_list[1])
    # print(TEST_list[1])
    # print('\n')
    max_epoch = max(TRAIN_list[0])

    plot_data = [[],[],[]]
    # print(type(plot_data))

    for i in range(0,max_epoch+2):
        if i in TRAIN_list[0]:
            plot_data[0].append(TRAIN_list[1][TRAIN_list[0].index(i)])
        else:
            plot_data[0].append(None)
        
        if i in VALID_list[0]:
            plot_data[1].append(VALID_list[1][VALID_list[0].index(i)])
        else:
            plot_data[1].append(None)
        
        if i in TEST_list[0]:
            plot_data[2].append(TEST_list[1][TEST_list[0].index(i)])
        else:
            plot_data[2].append(None)


    return plot_data

def plot_data(plot_data_list, plot_name):
    fig = plt.figure()


    ax = fig.add_subplot(1,1,1)

    xs = np.arange(len(plot_data_list[0]))
    TRAIN = np.array(plot_data_list[0]).astype(np.double)
    TRAIN_mask = np.isfinite(TRAIN)
    VALID = np.array(plot_data_list[1]).astype(np.double)
    VALID_mask = np.isfinite(VALID)
    TEST = np.array(plot_data_list[2]).astype(np.double)
    TEST_mask = np.isfinite(TEST)
    
    ax.plot(xs[TRAIN_mask], TRAIN[TRAIN_mask], linestyle='-', marker='o',alpha=0.5)
    ax.plot(xs[VALID_mask], VALID[VALID_mask], linestyle='-', marker='o',alpha=0.5)
    ax.plot(xs[TEST_mask], TEST[TEST_mask], linestyle='-', marker='o',alpha=0.5)
    
    ax.grid(True)
    # ax.set_ylim(bottom=0)
    ax.set_xlabel('epoch')  # x軸ラベル
    ax.set_ylabel(plot_name)  # y軸ラベル
    ax.set_title(plot_name)  # グラフタイトル
    ax.legend(['TRAIN','VALID','TEST'])
    
    save_name = new_dir_path+'/'+plot_name+'_'+file_log+'png'
    fig.savefig(save_name)
    
    plt.show()

def plot_data_microavg(plot_data_list, plot_name):
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)

    xs = np.arange(len(plot_data_list[0]))
    TRAIN = np.array(plot_data_list[0]).astype(np.double)
    TRAIN_mask = np.isfinite(TRAIN)
    VALID = np.array(plot_data_list[1]).astype(np.double)
    VALID_mask = np.isfinite(VALID)
    TEST = np.array(plot_data_list[2]).astype(np.double)
    TEST_mask = np.isfinite(TEST)
    
    ax.plot(xs[TRAIN_mask], TRAIN[TRAIN_mask], linestyle='-', marker='o',alpha=0.5)
    ax.plot(xs[VALID_mask], VALID[VALID_mask], linestyle='-', marker='o',alpha=0.5)
    ax.plot(xs[TEST_mask], TEST[TEST_mask], linestyle='-', marker='o',alpha=0.5)
    
    ax.grid(True)
    # ax.set_ylim(bottom=0)
    ax.set_xlabel('epoch')  # x軸ラベル
    ax.set_ylabel(plot_name)  # y軸ラベル
    ax.set_title(plot_name)  # グラフタイトル
    ax.legend(['TRAIN','VALID','TEST'])
    
    save_name = new_dir_path+'/microavg/'+plot_name+'_'+file_log+'png'
    fig.savefig(save_name)
    
    plt.show()

def plot_data_class0(plot_data_list, plot_name):
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)

    xs = np.arange(len(plot_data_list[0]))
    TRAIN = np.array(plot_data_list[0]).astype(np.double)
    TRAIN_mask = np.isfinite(TRAIN)
    VALID = np.array(plot_data_list[1]).astype(np.double)
    VALID_mask = np.isfinite(VALID)
    TEST = np.array(plot_data_list[2]).astype(np.double)
    TEST_mask = np.isfinite(TEST)
    
    ax.plot(xs[TRAIN_mask], TRAIN[TRAIN_mask], linestyle='-', marker='o',alpha=0.5)
    ax.plot(xs[VALID_mask], VALID[VALID_mask], linestyle='-', marker='o',alpha=0.5)
    ax.plot(xs[TEST_mask], TEST[TEST_mask], linestyle='-', marker='o',alpha=0.5)
    
    ax.grid(True)
    # ax.set_ylim(bottom=0)
    ax.set_xlabel('epoch')  # x軸ラベル
    ax.set_ylabel(plot_name)  # y軸ラベル
    ax.set_title(plot_name)  # グラフタイトル
    ax.legend(['TRAIN','VALID','TEST'])
    
    save_name = new_dir_path+'/class0/'+plot_name+'_'+file_log+'png'
    fig.savefig(save_name)
    
    plt.show()

def plot_data_class1(plot_data_list, plot_name):
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)

    xs = np.arange(len(plot_data_list[0]))
    TRAIN = np.array(plot_data_list[0]).astype(np.double)
    TRAIN_mask = np.isfinite(TRAIN)
    VALID = np.array(plot_data_list[1]).astype(np.double)
    VALID_mask = np.isfinite(VALID)
    TEST = np.array(plot_data_list[2]).astype(np.double)
    TEST_mask = np.isfinite(TEST)
    
    ax.plot(xs[TRAIN_mask], TRAIN[TRAIN_mask], linestyle='-', marker='o',alpha=0.5)
    ax.plot(xs[VALID_mask], VALID[VALID_mask], linestyle='-', marker='o',alpha=0.5)
    ax.plot(xs[TEST_mask], TEST[TEST_mask], linestyle='-', marker='o',alpha=0.5)
    
    ax.grid(True)
    # ax.set_ylim(bottom=0)
    ax.set_xlabel('epoch')  # x軸ラベル
    ax.set_ylabel(plot_name)  # y軸ラベル
    ax.set_title(plot_name)  # グラフタイトル
    ax.legend(['TRAIN','VALID','TEST'])
    
    save_name = new_dir_path+'/class1/'+plot_name+'_'+file_log+'png'
    fig.savefig(save_name)
    
    plt.show()

error_plot_data = make_plot_data(TRAIN_errors,VALID_errors,TEST_errors)
plot_data(error_plot_data,'ERROR')

loss_plot_data = make_plot_data(TRAIN_losses,VALID_losses,TEST_losses)
plot_data(loss_plot_data,'LOSS')

MAP_plot_data = make_plot_data(TRAIN_MAP,VALID_MAP,TEST_MAP)
plot_data(MAP_plot_data,'MAP')

MRR_plot_data = make_plot_data(TRAIN_MRR,VALID_MRR,TEST_MRR)
plot_data(MRR_plot_data,'MRR')

# microavg
microavg_pre_plot_data = make_plot_data(TRAIN_microavg_pre,VALID_microavg_pre,TEST_microavg_pre)
plot_data_microavg(microavg_pre_plot_data,'microavg_precision')

microavg_rec_plot_data = make_plot_data(TRAIN_microavg_rec,VALID_microavg_rec,TEST_microavg_rec)
plot_data_microavg(microavg_rec_plot_data,'microavg_recall')

microavg_f1_plot_data = make_plot_data(TRAIN_microavg_f1,VALID_microavg_f1,TEST_microavg_f1)
plot_data_microavg(microavg_f1_plot_data,'microavg_f1')

# class0
class0_pre_plot_data = make_plot_data(TRAIN_class0_pre,VALID_class0_pre,TEST_class0_pre)
plot_data_class0(class0_pre_plot_data,'class0_precision')

class0_rec_plot_data = make_plot_data(TRAIN_class0_rec,VALID_class0_rec,TEST_class0_rec)
plot_data_class0(class0_rec_plot_data,'class0_recall')

class0_f1_plot_data = make_plot_data(TRAIN_class0_f1,VALID_class0_f1,TEST_class0_f1)
plot_data_class0(class0_f1_plot_data,'class0_f1')


# class1
class1_pre_plot_data = make_plot_data(TRAIN_class1_pre,VALID_class1_pre,TEST_class1_pre)
plot_data_class1(class1_pre_plot_data,'class1_precision')

class1_rec_plot_data = make_plot_data(TRAIN_class1_rec,VALID_class1_rec,TEST_class1_rec)
plot_data_class1(class1_rec_plot_data,'class1_recall')

class1_f1_plot_data = make_plot_data(TRAIN_class1_f1,VALID_class1_f1,TEST_class1_f1)
plot_data_class1(class1_f1_plot_data,'class1_f1')