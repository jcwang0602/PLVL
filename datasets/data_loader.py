# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os.path as osp
import re
import sys
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.utils.data as data

sys.path.append(".")

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils.word_utils import Corpus


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer, is_train):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0 : (seq_length - 2)]
        tokens = []
        input_mask = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        input_mask.append(1)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
            if False and is_train and len(tokens_a) > 3 and np.random.random() > 0.8:
                input_mask.append(0)
            else:
                input_mask.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_mask.append(1)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(InputFeatures(unique_id=example.unique_id, tokens=tokens, input_ids=input_ids, input_mask=input_mask, input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass


class EEVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        "referit": {"splits": ("train", "val", "trainval", "test")},
        "unc": {"splits": ("train", "val", "trainval", "testA", "testB"), "params": {"dataset": "refcoco", "split_by": "unc"}},
        "unc+": {"splits": ("train", "val", "trainval", "testA", "testB"), "params": {"dataset": "refcoco+", "split_by": "unc"}},
        "gref": {"splits": ("train", "val"), "params": {"dataset": "refcocog", "split_by": "google"}},
        "gref_umd": {"splits": ("train", "val", "test"), "params": {"dataset": "refcocog", "split_by": "umd"}},
        "flickr": {"splits": ("train", "val", "test")},
        "mixed_pretrain": {"splits": ("train", "val")},
        "mixed_coco": {"splits": ("train", "val")},
    }

    def __init__(
        self,
        data_root,
        split_root="data",
        dataset="referit",
        transform=None,
        return_idx=False,
        testmode=False,
        split="train",
        max_query_len=128,
        lstm=False,
        bert_model="bert-base-uncased",
        is_segment=False,
    ):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx = return_idx
        self.is_segment = is_segment

        assert self.transform is not None

        if split == "train":
            self.augment = True
        else:
            self.augment = False

        if self.dataset == "referit":
            self.dataset_root = osp.join(self.data_root, "referit")
            self.im_dir = osp.join(self.dataset_root, "images")
            self.split_dir = osp.join(self.dataset_root, "splits")
        elif self.dataset == "flickr":
            self.dataset_root = osp.join(self.data_root, "Flickr30k")
            self.im_dir = osp.join(self.dataset_root, "flickr30k_images")
        else:
            self.dataset_root = osp.join(self.data_root, "mscoco")
            self.im_dir = osp.join(self.dataset_root, "images", "train2014")
            self.split_dir = osp.join(self.dataset_root, "splits")

        if not self.exists_dataset():
            print("Please put the dataset in the right place.")
            print(osp.join(self.split_root, self.dataset))
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]["splits"]

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, "corpus.pth")
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError("Dataset {0} does not have split {1}".format(self.dataset, split))

        splits = [split]
        if self.dataset != "referit":
            splits = ["train", "val"] if split == "trainval" else [split]
        for split in splits:
            imgset_file = "{0}_{1}.pth".format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path, weights_only=False)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == "flickr":
            img_file, bbox, phrase = self.images[idx]
            data_source = None
        else:
            img_file, data_source, bbox, phrase, attri = self.images[idx]

        if data_source == "coco":
            img_path = osp.join(self.im_dir, img_file)
        elif data_source == "flickr":
            img_path = osp.join(self.data_root, "flickr30k", img_file)
        elif data_source == "vg":
            img_path = osp.join(self.data_root, "visual-genome", img_file)
        else:
            img_path = osp.join(self.im_dir, img_file)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        gt_mask = None

        if self.is_segment:
            assert len(bbox) >= 5
            gt_mask_poly_vertices = bbox[4:]
            rles = maskUtils.frPyObjects(gt_mask_poly_vertices, h, w)
            is_crowd = 0
            if len(rles) > 1:
                is_crowd = 1
            # sometimes there are multiple binary map (corresponding to multiple segs)
            rle = maskUtils.merge(rles)
            gt_mask = maskUtils.decode(rle)
            gt_mask = Image.fromarray(gt_mask)
            bbox = bbox[:4]
        if len(bbox) >= 5:
            bbox = bbox[:4]
        # box format: to x1y1x2y2
        if not (self.dataset == "referit" or self.dataset == "flickr"):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        if self.is_segment:
            return img, phrase, bbox, gt_mask
        return img, phrase, bbox

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.is_segment:
            img, phrase, bbox, gt_mask = self.pull_item(idx)
        else:
            img, phrase, bbox = self.pull_item(idx)
        phrase = phrase.lower()
        input_dict = {"img": img, "box": bbox, "text": phrase}
        if self.is_segment:
            input_dict["gt_mask"] = gt_mask
        input_dict = self.transform(input_dict)
        img = input_dict["img"]
        bbox = input_dict["box"]
        phrase = input_dict["text"]
        img_mask = input_dict["mask"]
        if self.is_segment:
            gt_mask = input_dict["gt_mask"]

        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id > 0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer, is_train=(self.split == "train"))
            word_id = features[0].input_ids
            word_mask = features[0].input_mask

        if self.is_segment:
            return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32), gt_mask
        return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
