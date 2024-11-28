import torch

"""
束搜索是一种启发式搜索算法，可以在生成序列时保持多个候选序列，从而提高生成结果的质量。
"""
class Beam:

    """
    beam_size: 束的大小，决定在每一步保留多少个候选序列。
    min_length: 生成序列的最小长度。
    n_top: 需要返回的最佳序列数量。
    ranker: 一个可选的评分函数，用于对生成的序列进行排序。
    start_token_id 和 end_token_id: 序列开始和结束的标记。
    """
    def __init__(self, beam_size=8, min_length=0, n_top=1, ranker=None,
                 start_token_id=2, end_token_id=3):
        self.beam_size = beam_size
        self.min_length = min_length
        self.n_top = n_top
        self.ranker = ranker
        self.end_token_id = end_token_id

        # TODO:以下两个属性的含义
        self.top_sentence_ended = False
        self.prev_ks = []
        
        # 存储当前生成的序列，初始化为包含开始标记的张量。
        self.next_ys = [torch.LongTensor(beam_size).fill_(start_token_id)]

        # 存储当前候选序列的得分。
        self.current_scores = torch.FloatTensor(beam_size).zero_()
        
        # self.all_scores: 用于保存每个时间步的所有得分。
        self.all_scores = []

        # 存储已完成的序列及其得分。
        self.finished = []
       


    def advance(self, next_log_probs):
        """根据给定的下一个对数概率 next_log_probs 更新束搜索的状态。"""
        # next_probs : beam_size X vocab_size
        
        vocabulary_size = next_log_probs.size(-1)
        # current_beam_size = next_log_probs.size(0)

        # 处理 EOS（结束标记）逻辑，确保在生成的序列长度小于最小长度时，不允许生成结束标记。
        current_length = len(self.next_ys)
        if current_length < self.min_length:
            for beam_index in range(len(next_log_probs)):
                next_log_probs[beam_index][self.end_token_id] = -1e10

        # 计算每个候选序列的得分，保留 beam_size 个得分最高的序列。
        if len(self.prev_ks) > 0:
            next_log_probs = next_log_probs[:,-1,:]
            beam_scores = next_log_probs + self.current_scores.unsqueeze(1).expand_as(next_log_probs)
            # Don't let EOS have children.
            last_y = self.next_ys[-1]
            for beam_index in range(last_y.size(0)):
                if last_y[beam_index] == self.end_token_id:
                    beam_scores[beam_index] = -1e10 # -1e20 raises error when executing
        else:
            beam_scores = next_log_probs[0]     # 在beam_size维度取第一个，本质上都是重复的
            
        flat_beam_scores = beam_scores.view(-1)
        top_scores, top_score_ids = flat_beam_scores.topk(k=self.beam_size, dim=0, largest=True, sorted=True)   # 当前时间步得分最高的beam_size个候选词得分及其在词表中的索引

        self.current_scores = top_scores
        self.all_scores.append(self.current_scores)
        
        # 使用 prev_ks 和 next_ys 记录当前时间步的状态，并收集注意力信息。
        prev_k = top_score_ids // vocabulary_size  # (beam_size, )
        next_y = top_score_ids - prev_k * vocabulary_size  # (beam_size, )

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)

        # 如果某个序列达到了结束标记，则将其添加到 finished 列表中。
        for beam_index, last_token_id in enumerate(next_y):
            if last_token_id == self.end_token_id:
                # skip scoring
                self.finished.append((self.current_scores[beam_index], len(self.next_ys) - 1, beam_index))

        if next_y[0] == self.end_token_id:
            self.top_sentence_ended = True

    def get_current_state(self):
        "Get the outputs for the current timestep.获取当前时间步生成的序列。"
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep.获取当前时间步的 backpointer（回指针），用于追踪生成序列"
        return self.prev_ks[-1]

    def done(self):
        "检查是否完成了生成过程。条件是顶级序列已经结束并且已完成的序列数量达到 n_top"
        return self.top_sentence_ended and len(self.finished) >= self.n_top

    def get_hypothesis(self, timestep, k):
        "根据指定的时间步和序列索引 k 返回生成的序列和对应的注意力矩阵。"
        hypothesis = []
        
        # 反向遍历状态，收集生成的序列和注意力信息。
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hypothesis.append(self.next_ys[j + 1][k])
            # for RNN, [:, k, :], and for trnasformer, [k, :, :]
            k = self.prev_ks[j][k]
        return hypothesis[::-1]

    def sort_finished(self, minimum=None):
        "对完成的序列进行排序，并根据需要填充至少 minimum 个输出。"
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            "将完成的序列按照得分降序排序，返回得分和序列索引。"
            while len(self.finished) < minimum:
                # global_scores = self.global_scorer.score(self, self.scores)
                # s = global_scores[i]
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished = sorted(self.finished, key=lambda a: a[0], reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks