from beam import Beam
from utils.pad import pad_masking

import torch


class Predictor:

    def __init__(self, preprocess, postprocess, model, checkpoint_filepath, max_length=30, beam_size=8):
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.model = model
        self.max_length = max_length
        self.beam_size = beam_size

        self.model.eval()
        checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def predict_one_Beam(self, source, num_candidates=5):
        """
        source:源语句
        num_candidates:候选目标语句个数
        """
        source_preprocessed = self.preprocess(source)
        source_tensor = torch.tensor(source_preprocessed).unsqueeze(0)  # why unsqueeze?
        length_tensor = torch.tensor(len(source_preprocessed)).unsqueeze(0)

        sources_mask = pad_masking(source_tensor, source_tensor.size(1))
        memory_mask = pad_masking(source_tensor, 1)
        memory = self.model.encoder(source_tensor, sources_mask)    #(1, Ls, 512)

        decoder_state = self.model.decoder.init_decoder_state()

        # Repeat beam_size times
        memory_beam = memory.detach().repeat(self.beam_size, 1, 1)  # (beam_size, seq_len, hidden_size)

        # TODO:Beam作用是什么？
        beam = Beam(beam_size=self.beam_size, min_length=0, n_top=num_candidates, ranker=None)  # 重复beam_size次
        global old_inputs
        for step in range(self.max_length):

            new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
            
            if (step!=0):
                new_inputs = torch.cat([old_inputs, new_inputs], dim=1)
            decoder_outputs, decoder_state = self.model.decoder(new_inputs, memory_beam,
                                                                            memory_mask,
                                                                            state=decoder_state)
            # decoder_outputs: (beam_size, target_seq_len=1, vocabulary_size)
            # attentions['std']: (target_seq_len=1, beam_size, source_seq_len)

            beam.advance(decoder_outputs.squeeze(1))

            old_inputs = new_inputs
            beam_current_origin = beam.get_current_origin()  # (beam_size, )
            decoder_state.beam_update(beam_current_origin)

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=num_candidates)
        hypothesises = []
        for i, (times, k) in enumerate(ks[:num_candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

        self.hypothesises = [[token.item() for token in h] for h in hypothesises]
        hs = [self.postprocess(h) for h in self.hypothesises]
        return list(reversed(hs))
    
    def predict_one_Greedy(self, source):
        """
        source:源语句
        num_candidates:候选目标语句个数
        """
        source_preprocessed = self.preprocess(source)
        source_tensor = torch.tensor(source_preprocessed).unsqueeze(0)  # why unsqueeze?
        

        sources_mask = pad_masking(source_tensor, source_tensor.size(1))
        memory_mask = pad_masking(source_tensor, 1)
        memory = self.model.encoder(source_tensor, sources_mask)    #(1, Ls, 512)

        decoder_state = self.model.decoder.init_decoder_state()

        new_inputs = torch.LongTensor([[2]])
        pred_result = []
        
        # 不超过最长序列的情况下一直迭代；采取贪心搜索策略，未考虑全局
        for _ in range(self.max_length):
            decoder_outputs, decoder_state = self.model.decoder(new_inputs, memory, memory_mask, state=decoder_state)
            decoder_outputs = decoder_outputs[:,-1:,]   # 只取最新预测出来的单词的概率
            _, next_word = torch.max(decoder_outputs, dim=2)
            pred_result.append(next_word.detach().item())
            
            if(next_word == 3): # 遇到终止符则停止
                break
            
            new_inputs = torch.cat([new_inputs, next_word], dim=1)
        hs = [self.postprocess(h) for h in [pred_result]]
        return list(reversed(hs))
    
    