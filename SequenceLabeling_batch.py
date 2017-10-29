from __future__ import division

# -*- coding: utf-8 -*-
__author__ = 'huangyf'
import numpy as np
import pandas as pd
import math
import re
import pickle
import random
import time
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from masked_cross_entropy import masked_cross_entropy

# from warpctc_pytorch import CTCLoss

train_labels_path = "./assignment1fall/train.lab"
train_frames_path = "./assignment1fall/mfcc/train.ark"
test_frames_path = "./assignment1fall/mfcc/test.ark"
map_48to39_path = "./assignment1fall/48_39.map"
phone2alpha_path = "./assignment1fall/48phone_char.map"

phone2alpha = dict(np.loadtxt(phone2alpha_path, str)[:, [True, False, True]].tolist())


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s ' % (asMinutes(s))


def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)
    if len(target) == 0:
        return len(source)
    source = np.array(tuple(source))
    target = np.array(tuple(target))
    previous_row = np.arange(target.size + 1)
    for s in source:
        current_row = previous_row + 1
        current_row[1:] = np.minimum(
            current_row[1:],
            np.add(previous_row[:-1], target != s))
        current_row[1:] = np.minimum(
            current_row[1:],
            current_row[0:-1] + 1)
        previous_row = current_row
    return previous_row[-1]


def frame2phone(frame):
    phone = []
    for i in range(len(frame)):
        if i < 1:
            phone.append(frame[i])
        else:
            if frame[i - 1] != frame[i]:
                phone.append(frame[i])
    if phone[0] == 'L':
        phone.pop(0)
    if len(phone) > 0:
        if phone[-1] == 'L' or phone[-1] == '<PAD>':
            phone.pop(-1)
    if len(phone) > 0:
        if phone[-1] == 'L' or phone[-1] == '<PAD>':
            phone.pop(-1)
    return phone


def preprocessor(frames_path, labels_path=None, map_48to39_path=None):
    print('Start to process the data......')
    start = time.time()
    frames = np.loadtxt(frames_path, str)
    frames_name = frames[:, 0].tolist()
    names = pd.DataFrame(frames_name)
    names[1] = names.applymap(lambda x: int(re.findall(r'_([0-9]+)', x)[0]))
    names[0] = names[0].apply(lambda x: re.findall(r'(\w+_\w+)_', x)[0])
    name_unique = names[0].unique().tolist()
    frames_num = names.groupby([0]).max()[1].tolist()
    frames = frames[:, 1:].astype(np.float32)
    frames_1 = [frames[:frames_num[0], :]]
    frames = [frames[sum(frames_num[:i]):sum(frames_num[:(i + 1)]), :] for i in range(len(frames_num))]
    frames += frames_1
    if labels_path is not None:
        labels = np.loadtxt(labels_path, str)
        labels = list(map(lambda x: x.split(','), labels))
        labels_name, labels = list(zip(*labels))
        labels_dict = dict(list(zip(labels_name, labels)))
        if map_48to39_path:
            map_48to39 = dict(np.loadtxt(map_48to39_path, str).tolist())
            labels = [phone2alpha[map_48to39[labels_dict[name]]] for name in frames_name]
        else:
            labels = [phone2alpha[labels_dict[name]] for name in frames_name]
        labels_1 = [labels[:frames_num[0]]]
        labels = [labels[sum(frames_num[:i]):sum(frames_num[:(i + 1)])] for i in range(len(frames_num))]
        labels += labels_1
        print('Data processing is finished, time={:.6f}s'.format(time.time() - start))
        return frames_name, name_unique, frames_num, frames, labels
    else:
        print('Data processing is finished, time is {:.6f}s'.format(time.time() - start))
        return frames_name, name_unique, frames_num, frames


def normalnize(frames):
    mean = np.mean(frames, axis=1)
    var = np.sqrt(np.var(frames, axis=1))
    return (frames - mean.reshape(-1, 1)) / var.reshape(-1, 1)


def LongTensor(one_list):
    return Variable(torch.LongTensor([one_list])).cuda()


def FloatTensor(one_list):
    return Variable(torch.FloatTensor([one_list])).cuda()


def Pick(frames_num, frames, labels, epoch=None):
    if epoch != None:
        idx = epoch % len(frames_num)
    else:
        idx = random.choice(range(len(frames_num)))
    return frames[sum(frames_num[:idx]):sum(frames_num[:(idx + 1)]), :], \
           labels[sum(frames_num[:idx]):sum(frames_num[:(idx + 1)])]


def PickBatch(frames, labels, frames_length, batch_size):
    frames_batch = []
    labels_batch = []
    length_batch = []
    for i in range(batch_size):
        idx = random.choice(range(len(frames)))
        frames_batch.append(frames[idx])
        labels_batch.append(labels[idx])
        length_batch.append(frames_length[idx])
    batch = sorted(zip(frames_batch, labels_batch), key=lambda p: len(p[0]), reverse=True)
    frames_batch, labels_batch = zip(*batch)
    labels_batch = [label[:max(length_batch)] for label in labels_batch]
    return frames_batch, labels_batch, sorted(length_batch, reverse=True)


def PickBatchSeq(frames, labels, frames_length, batch_size):
    n = int(len(frames) / batch_size)
    for i in range(n):
        batch = sorted(zip(frames[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size]),
                       key=lambda p: len(p[0]), reverse=True)
        frames_batch, labels_batch = zip(*batch)
        length_batch = frames_length[i * batch_size:(i + 1) * batch_size]
        labels_batch = [label[:max(length_batch)] for label in labels_batch]
        yield frames_batch, labels_batch, sorted(length_batch, reverse=True)


def train_val_split(frames, frames_num, labels, train=0.8):
    n = len(frames)
    idx = int(train * n)
    print('train_size:{}, val_size:{}'.format(idx, n - idx))
    return frames[:idx], frames_num[:idx], labels[:idx], \
           frames[idx:], frames_num[idx:], labels[idx:]


## padding to batch
def pad(frames, frames_num, max_length, labels=None):
    e = labels is not None
    for i in range(len(frames_num)):
        l, m = frames[i].shape
        if l >= max_length:
            frames[i] = frames[i][:max_length, :]
            if e:
                labels[i] = labels[i][:max_length]
        else:
            frames[i] = np.concatenate((frames[i], np.zeros((max_length - l, m))), axis=0)
            if e:
                labels[i] = labels[i] + [label2idx['<PAD>']] * (max_length - l)
    if e:
        return frames, labels
    else:
        return frames


def Batch2Tensor(frames_batch, labels_batch):
    frames_tensor = Variable(torch.from_numpy(np.array(frames_batch)).float()).cuda()
    labels_tensor = Variable(torch.LongTensor(list(labels_batch))).cuda()
    return frames_tensor, labels_tensor


def evaluate(predict_batch, length_batch, labels_batch):
    error_lv = []
    error = []
    batch_size = predict_batch.shape[0]
    for j in range(batch_size):
        prediction_one = [idx2label[idx] for idx in predict_batch[j, :length_batch[j]]]
        prediction_phone = ''.join(frame2phone(prediction_one))
        true_one = [idx2label[idx] for idx in labels_batch[j][:length_batch[j]]]
        true_phone = ''.join(frame2phone(true_one))
        error_lv.append(levenshtein(true_phone, prediction_phone))
        error.append(np.sum(np.array(prediction_one) != np.array(true_one)) / length_batch[j])
    return error_lv, error


# frames_name, name_unique, frames_num, frames, labels = preprocessor(train_frames_path, train_labels_path,map_48to39_path)
# frames_name_test, name_unique_test, frames_num_test, frames_test = preprocessor(test_frames_path)
frames_name = pickle.load(open('frames_name.pkl', 'rb'))
name_unique = pickle.load(open('name_unique.pkl', 'rb'))
frames_num = pickle.load(open('frames_num.pkl', 'rb'))
frames = pickle.load(open('frames.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))
frames_name_test = pickle.load(open('frames_name_test.pkl', 'rb'))
name_unique_test = pickle.load(open('name_unique_test.pkl', 'rb'))
frames_num_test = pickle.load(open('frames_num_test.pkl', 'rb'))
frames_test = pickle.load(open('frames_test.pkl', 'rb'))

# frames=normalnize(frames)
# frames_test=normalnize(frames_test)

n_classes = 40  # 39+1
train_size_all = frames[0].shape[1]
num_frame = len(frames)
lr = 1e-2
batch_size = 32
hidden_size = 128
hidden_size_cnn = 32
num_epoch = 15
layer_size = 1
kernel_size = 31
stride = 1
padding = 15
max_length = 800

map_48to39 = dict(np.loadtxt(map_48to39_path, str).tolist())
alphas = [phone2alpha[i] for i in map_48to39.values()]

label2idx, idx2label = {}, {}
n = 0
for label in alphas:
    if label not in label2idx.keys():
        label2idx[label] = n
        idx2label[n] = label
        n += 1
label2idx['<PAD>'] = n
idx2label[n] = '<PAD>'

labels = [[label2idx[i] for i in label] for label in labels]

frames_train, frames_num_train, labels_train, \
frames_val, frames_num_val, labels_val = train_val_split(frames, frames_num, labels, train=0.9)

train_size = len(frames_train)
val_size = len(frames_val)
pad_val_size = batch_size - val_size % batch_size
frames_val.extend([frames_val[-1]] * pad_val_size)
frames_num_val.extend([frames_num_val[-1]] * pad_val_size)
labels_val.extend([labels_val[-1]] * pad_val_size)
print('val_extend_size:{}'.format(pad_val_size))

frames_pad, labels_pad = pad(frames_train, frames_num_train, max_length, labels_train)
frames_pad_val, labels_pad_val = pad(frames_val, frames_num_val, max_length, labels_val)


# frames_test_pad = pad(frames_test, frames_num_test, max_length)


class cnn(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, drop_out):
        super(cnn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cnn_ = nn.Conv1d(self.input_size, self.output_size, kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding)
        self.drop_out = nn.Dropout(p=drop_out)
        self.batch_norm = nn.BatchNorm1d(self.output_size)

        nn.init.xavier_normal(self.cnn_.weight)

    def forward(self, frames):
        frames_cnn = self.cnn_(frames)
        frames_cnn = self.batch_norm(frames_cnn)
        frames_cnn = self.drop_out(frames_cnn)
        frames_cnn = self.cnn_(frames_cnn)
        frames_cnn = self.batch_norm(frames_cnn)
        frames_cnn = self.drop_out(frames_cnn)
        # frames_cnn = self.cnn_(frames_cnn)
        # frames_cnn = self.drop_out(frames_cnn)
        return frames_cnn


class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_size, drop_out, bidirectional=True):
        super(lstm, self).__init__()
        self.bidirectional = bidirectional
        self.bidirectional_number = 2 if bidirectional else 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_out = drop_out
        self.layer_size = layer_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=self.layer_size, batch_first=True,
                          bidirectional=self.bidirectional)
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.layer_size, batch_first=True,
                             bidirectional=self.bidirectional)
        self.lstm2 = nn.LSTM(self.hidden_size * 2, self.hidden_size, num_layers=self.layer_size, batch_first=True,
                             bidirectional=self.bidirectional)
        self.lstm3 = nn.LSTM(self.hidden_size * 2, self.hidden_size, num_layers=self.layer_size, batch_first=True,
                             bidirectional=self.bidirectional)
        self.drop = nn.Dropout(drop_out)
        self.linear = nn.Linear(self.hidden_size * self.bidirectional_number, self.output_size)
        self.batch_norm = nn.BatchNorm2d(max_length)

    def forward(self, frame, hidden, frame_length):
        frame = self.batch_norm(frame)
        output, output_length, hidden = self.train_lstm(frame, frame_length, self.lstm1, hidden)
        output = self.drop(output)

        output, output_length, hidden = self.train_lstm(output, output_length, self.lstm2, hidden)
        output = self.drop(output)

        output, output_length, hidden = self.train_lstm(output, output_length, self.lstm3, hidden)
        output = self.drop(output)

        output, output_length, hidden = self.train_lstm(output, output_length, self.lstm3, hidden)
        output = self.drop(output)

        output = self.linear(output)
        return output

    def train_lstm(self, input, input_length, method, hidden):
        packed = pack_padded_sequence(input, input_length, batch_first=True)
        packed, hidden = method(packed, hidden)
        output, output_length = pad_packed_sequence(packed, batch_first=True)
        return output, output_length, hidden

    def initHidden(self):
        h = Variable(torch.zeros(self.layer_size * self.bidirectional_number, batch_size, self.hidden_size)).cuda()
        c = Variable(torch.zeros(self.layer_size * self.bidirectional_number, batch_size, self.hidden_size)).cuda()
        return (h, c)


if __name__ == '__main__':

    # cnn_model = cnn(max_length,max_length,kernel_size=kernel_size, stride=stride, padding=padding, drop_out=0.1).cuda()
    model = lstm(train_size_all, hidden_size=hidden_size, output_size=n_classes, layer_size=layer_size, drop_out=0.3,
                 bidirectional=True).cuda()
    # criterion = nn.NLLLoss()
    # criterion=CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    # optimizer=optim.SGD(model.parameters(),lr,momentum=0.9,nesterov=True)
    # optimizer = optim.RMSprop(model.parameters(), lr, momentum=0.9)
    # optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=lr)
    #
    print('Start to train.....')

    start = time.time()
    loss_ = []
    loss_lv = []
    accuracy = []
    loss_lv_val = []
    accuracy_val = []
    for epoch in range(num_epoch):
        batch_epoch = int(len(frames_train) / batch_size)
        for i in range(batch_epoch):
            # model.train()
            frames_batch, labels_batch, length_batch = PickBatch(frames_pad, labels_pad, frames_num_train, batch_size)
            frames_tensor, labels_tensor = Batch2Tensor(frames_batch, labels_batch)
            hidden = model.initHidden()
            # optimizer_cnn.zero_grad()
            optimizer.zero_grad()
            # output=cnn_model.forward(frames_tensor)
            output = model.forward(frames_tensor, hidden, length_batch)
            loss = masked_cross_entropy(output.contiguous(), labels_tensor.contiguous(), length_batch)
            loss.backward()
            # optimizer_cnn.step()
            optimizer.step()
            predict_batch = output.topk(1)[1].cpu().data[:, :, 0].numpy()
            error_lv, error = evaluate(predict_batch, length_batch, labels_batch)
            accuracy.append(1 - sum(error) / batch_size)
            loss_lv.append(sum(error_lv) / batch_size)
            loss_.append(loss.data[0])

            # model.eval()
            # accuracy_val_tmp = []
            # loss_lv_val_tmp = []
            # for frames_batch, labels_batch, length_batch in PickBatchSeq(frames_pad_val, labels_pad_val, frames_num_val,
            #                                                              batch_size):
            #     frames_tensor, labels_tensor = Batch2Tensor(frames_batch, labels_batch)
            #     hidden = model.initHidden()
            #     output = model.forward(frames_tensor, hidden, length_batch)
            #     predict_batch = output.topk(1)[1].cpu().data[:, :, 0].numpy()
            #     error_lv, error = evaluate(predict_batch, length_batch, labels_batch)
            #     accuracy_val_tmp.extend(error)
            #     loss_lv_val_tmp.extend(loss_lv)
            # accuracy_val.append(1 - sum(accuracy_val_tmp[:val_size]) / val_size)
            # loss_lv_val.append(sum(loss_lv_val_tmp[:val_size]) / val_size)
            # if i % 10 == 0:
            #     print('time:{} epoch{} [{}/{}]---loss:{:.5f}---train_lv_loss:{:.4f}---'
            #           'train_acc:{:.4f}%\nval_lv_loss:{:.4f}---val_acc:{:.4f}%'.format(timeSince(start), epoch + 1, i,
            #                                                                            batch_epoch,
            #                                                                            loss.data[0], loss_lv[-1],
            #                                                                            accuracy[-1] * 100,
            #                                                                            loss_lv_val[-1],
            #                                                                            accuracy_val[-1] * 100))
            if i % 10 == 0:
                print('time:{} epoch{} [{}/{}]---loss:{:.5f}---train_lv_loss:{:.4f}---'
                      'train_acc:{:.4f}%'.format(timeSince(start), epoch + 1, i, batch_epoch,
                                                 loss.data[0], loss_lv[-1], accuracy[-1] * 100))
    torch.save(model.state_dict(), 'model_param.pkl')
    np.savetxt('train_acc.txt', accuracy)
    np.savetxt('train_lv_loss.txt', loss_lv)
    # np.savetxt('val_acc.txt', accuracy_val)
    # np.savetxt('val_lv_loss.txt', loss_lv_val)
    # cnn_model.eval()
    # model.load_state_dict(torch.load('model_lstm_param.pkl'))
    # model.eval()
    # predictions_test = []

    # print('start to predict.....')
    # start = time.time()
    # for i in range(len(frames_test)):
    #     frame = Variable(torch.from_numpy(frames_test[i]).float()).cuda()
    #     hidden = model.initHidden(train=False)
    #     output = model.forward(frame, hidden[0], train=False)
    #     predict = output.topk(1)[1].cpu().data.numpy()
    #     prediction_one = [idx2label[idx[0]] for idx in predict]
    #     predictions_test.append(''.join(frame2phone(prediction_one)))
    # print('prediction finished.....')
    # with open('predictions_1.txt', 'a+') as f:
    #     for item in predictions_test:
    #         f.write('%s\n' % item)

    print('start to evalute....')
    model.eval()
    accuracy_val_tmp = []
    loss_lv_val_tmp = []
    for frames_batch, labels_batch, length_batch in PickBatchSeq(frames_pad_val, labels_pad_val, frames_num_val,
                                                                 batch_size):
        frames_tensor, labels_tensor = Batch2Tensor(frames_batch, labels_batch)
        hidden = model.initHidden()
        output = model.forward(frames_tensor, hidden, length_batch)
        predict_batch = output.topk(1)[1].cpu().data[:, :, 0].numpy()
        error_lv, error = evaluate(predict_batch, length_batch, labels_batch)
        accuracy_val_tmp.extend(error)
        loss_lv_val_tmp.extend(loss_lv)
    accuracy_val.append(1 - sum(accuracy_val_tmp[:val_size]) / val_size)
    loss_lv_val.append(sum(loss_lv_val_tmp[:val_size]) / val_size)
