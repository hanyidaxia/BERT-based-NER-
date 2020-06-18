import os
import sys
import math
import numpy as np
import math
from parser import *
from settings import *

class get_examples(object):

    def __init__(self, text):
        self.text = text

class h_reader(object):

    def __init__(self, batch_size):
        self.batch_size = args.batch_size

    def get_train_examples(self, data_dir):
        return self._data_importer(os.path.join(data_dir, 'train.txt'))
    def get_dev_examples(self, data_dir):
        return self._data_importer(os.path.join(data_dir, 'dev.txt'))
    def get_test_examples(self, data_dir):
        return self._data_importer(os.path.join(data_dir, 'new_test.txt'))
    def get_predict_examples(self, data_dir):
        return self._data_importer(os.path.join(data_dir, 'rest.txt'))


    def get_labels(self):
        return {'NA': 0, 'S':1, 'D':2, 'A':3}

    def _data_importer(self, in_file):
        print('reading file', in_file, file = SHELL_OUT_FILE, flush = True)
        contents =[]
        cnt = 0
        print('read lines', cnt, file = SHELL_OUT_FILE, flush = True)
        with open(in_file, 'r', encoding = 'UTF-8') as f:
            line = f.readline()
            while line:
                cnt += 1
                if cnt % 200 == 0:
                    print('\rRead line:', cnt//2,end = '\r', file = SHELL_OUT_FILE, flush = True)
                contents.append(line)
                line = f.readline()
            print('read over, there are d% lines' % cnt )
            # ntmp = copy.deepcopy(contents[0:2])
            # print("\rThe length of the contents is " + str(len(contents)) + '\n'
            #     "The type of the contents is" + str(type(contents)) + '\n'
            #     "The shape of the contents is" + str(np.shape(contents)) + '\n'
            #     "The 1st line of the contents is " + str(ntmp[0]) + '\n'
            #     "The 2nd line of the contents is " + str(ntmp[1]))

        examples = []
        t_examples = len(contents) //2

        n_text = []
        print("The type of the inputs is " + str(type(contents)) + "The shape of the inputs is " + str(np.shape(contents)))

        for i, juzi in enumerate(contents):
            if ((i+1)/2)/10 == 0:
                print('\rprocessed exapmles is : {}/{}'.format(((i+1)/2), t_examples), end = '\r', file = SHELL_OUT_FILE, flush = True)
            n_text.append(juzi)
            if ((i+1)/2)%self.batch_size == 0:
                # print('len', len(n_text))
                examples.append(get_examples(text = n_text))
                n_text = []
        if len(n_text):
                examples.append(get_examples(text = n_text))
        print("\rProcessed Examples: {}/{}".format(len(examples), t_examples), file=SHELL_OUT_FILE, flush=True)

        # ttmp = copy.deepcopy(examples[0:2])
        # print("\rThe length of the examples is " + str(len(examples)) + '\n'
        #     "The type of the examples is" + str(type(examples)) + '\n'
        #     "The shape of the examples is" + str(np.shape(examples)) + '\n'
        #     "The 1st line of the examples is " + str(ttmp[0]) + '\n'
        #     "The 2nd line of the examples is " + str(ttmp[-1]))
        return examples

class h_processor(object):

    def __init__(self, tokenizer, max_seq_len = args.max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len


    def processor(self, examples, label_types):
        length = len(examples.text)
        # print("cnmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm", np.shape(examples.text))
        input_id = np.zeros((length//2, self.max_seq_len), dtype = np.int64)
        token_type_id = np.zeros_like(input_id)
        labels = np.zeros((length//2, self.max_seq_len - 2), dtype = np.int64)

        for i, seq in enumerate(examples.text):
            if i & 1: # 取奇数
                continue
            # tokens = self.tokenizer.encode(seq, add_special_tokens = True)
            tokens = self.tokenizer.tokenize(seq)
            # print(tokens)
            # print("zhiqinaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", seq)
            # seq = seq.split
            # print("zhihouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu", seq)
            ori_lab = examples.text[i+1].split()
            # print("ori_labbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", ori_lab, len(ori_lab))
            target = []
            idx = 0
            for w in tokens:
                if w[:2] == "##":
                    target.append(target[-1])
                else:
                    # print(idx)
                    target.append(label_types[ori_lab[idx]])
                    idx += 1

            tokens = tokens[:(self.max_seq_len - 2)]
            target = target[:(self.max_seq_len - 2)]
            tokens = ['[CLS]'] + tokens + ['[SEP]']

            pad_len = self.max_seq_len - len(tokens)
            tokens.extend(['[PAD]'] * pad_len)
            segment_ids = [0] * len(tokens)
            target.extend([0] * (self.max_seq_len - 2 - len(target)))
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            input_id[i // 2, :] = tokens[:]

            # 百分之百确定这个inputid没问题了

            token_type_id[i // 2, :] = segment_ids[:]
            labels[i // 2, :] = target[:]
            # print("labelssssssssssssssssssssssssssssssssssssssssss", labels)



        """Following is the section that can """
        input_id = torch.from_numpy(input_id).long().detach()
        # print("tokensizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", np.shape(tokens))
        # print("inputidddddddddddddddddddddddddddddddddddd", np.shape(input_id), input_id)
        token_type_id = torch.from_numpy(token_type_id).long().detach()
        # print("tokentypeddddddddddddddddddddddddddddddddddddddddddddddddd", np.shape(token_type_id), token_type_id)
        inputs_mask = (input_id != 0).long().detach()
        # print("inputmasskkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk", np.shape(inputs_mask), inputs_mask)
        # inputs = [input_id, inputs_mask, token_type_id]
        inputs = [input_id, inputs_mask]
        labels = torch.from_numpy(labels).long().detach()
        # print("first_labelssssssssssssssssssssssssssssssssssssssssssssssssssssssssss", labels)

        if USE_CUDA:
            inputs = [i.cuda() for i in inputs]
            labels = labels.cuda()
        # print("second_labelssssssssssssssssssssssssssssssssssssssssssssssssssssssssss", labels)
        return inputs, labels


    def pred_processor(self, examples):
        length = len(examples.text)
        input_id = np.zeros((length//2, self.max_seq_len), dtype = np.int64)
        token_type_id = np.zeros_like(input_id)

        for i, seq in enumerate(examples.text):
            if i & 1: # 取奇数
                continue
            tokens = self.tokenizer.tokenize(seq)
            ori_lab = examples.text[i+1].split()
            tokens = tokens[:(self.max_seq_len - 2)]
            tokens = ['[CLS]'] + tokens + ['[SEP]']

            pad_len = self.max_seq_len - len(tokens)
            tokens.extend(['[PAD]'] * pad_len)
            segment_ids = [0] * len(tokens)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            input_id[i // 2, :] = tokens[:]
            # 百分之百确定这个inputid没问题了

            token_type_id[i // 2, :] = segment_ids[:]
        """Following is the section that can """
        input_id = torch.from_numpy(input_id).long().detach()
        inputs_mask = (input_id != 0).long().detach()
        inputs = [input_id, inputs_mask]

        if USE_CUDA:
            inputs = [i.cuda() for i in inputs]
        # print("second_labelssssssssssssssssssssssssssssssssssssssssssssssssssssssssss", labels)
        return inputs

    def convert_tensor_to_tokens(self, tensor):
        new_ids = []
        ids = tensor.cpu().numpy().tolist()
        for i in range(len(ids)):
            new_ids.append(self.tokenizer.convert_ids_to_tokens(ids[i]))
        return new_ids
