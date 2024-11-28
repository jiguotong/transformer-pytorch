from datasets import TranslationDataset, TranslationDatasetOnTheFly
from datasets import TokenizedTranslationDataset, TokenizedTranslationDatasetOnTheFly
from datasets import InputTargetTranslationDataset, InputTargetTranslationDatasetOnTheFly
from datasets import IndexedInputTargetTranslationDataset, IndexedInputTargetTranslationDatasetOnTheFly
from dictionaries import IndexDictionary
from utils.pipe import shared_tokens_generator, source_tokens_generator, target_tokens_generator

from argparse import ArgumentParser

parser = ArgumentParser('Prepare datasets')
parser.add_argument('--train_source', type=str, default='data/example/raw/src-train.txt')
parser.add_argument('--train_target', type=str, default='data/example/raw/tgt-train.txt')
parser.add_argument('--val_source', type=str, default='data/example/raw/src-val.txt')
parser.add_argument('--val_target', type=str, default='data/example/raw/tgt-val.txt')
parser.add_argument('--save_data_dir', type=str, default='data/example/processed')
parser.add_argument('--share_dictionary', type=bool, default=False)

args = parser.parse_args()

# 1.调整本地数据，source-target匹配，保存至本地
TranslationDataset.prepare(args.train_source, args.train_target, args.val_source, args.val_target, args.save_data_dir)
# translation_dataset = TranslationDataset(args.save_data_dir, 'train')       # 获得数据列表
# translation_dataset_on_the_fly = TranslationDatasetOnTheFly('train')
# assert translation_dataset[0] == translation_dataset_on_the_fly[0]

# 2.读取source-target数据，作为一个dataset实例
tokenized_dataset = TokenizedTranslationDataset(args.save_data_dir, 'train')

# 3.基于以上dataset进行统计，得到词表字典，每个词汇出现的次数，保存为vocabulary-mode.txt文件
if args.share_dictionary:
    source_generator = shared_tokens_generator(tokenized_dataset)
    source_dictionary = IndexDictionary(source_generator, mode='source')
    target_generator = shared_tokens_generator(tokenized_dataset)
    target_dictionary = IndexDictionary(target_generator, mode='target')

    source_dictionary.save(args.save_data_dir)
    target_dictionary.save(args.save_data_dir)
else:
    source_generator = source_tokens_generator(tokenized_dataset)
    source_dictionary = IndexDictionary(source_generator, mode='source')
    target_generator = target_tokens_generator(tokenized_dataset)
    target_dictionary = IndexDictionary(target_generator, mode='target')

    source_dictionary.save(args.save_data_dir)
    target_dictionary.save(args.save_data_dir)

# source_dictionary = IndexDictionary.load(args.save_data_dir, mode='source')
# target_dictionary = IndexDictionary.load(args.save_data_dir, mode='target')

# 4. 将sourcey以及target的每个token均换成其在词表中对应的idx，放在一行，分别是a b c d   StartSent aa bb cc dd   aa bb cc dd EndSent 
IndexedInputTargetTranslationDataset.prepare(args.save_data_dir, source_dictionary, target_dictionary)
# indexed_translation_dataset = IndexedInputTargetTranslationDataset(args.save_data_dir, 'train')
# indexed_translation_dataset_on_the_fly = IndexedInputTargetTranslationDatasetOnTheFly('train', source_dictionary, target_dictionary)
# assert indexed_translation_dataset[0] == indexed_translation_dataset_on_the_fly[0]

print('Done datasets preparation.')