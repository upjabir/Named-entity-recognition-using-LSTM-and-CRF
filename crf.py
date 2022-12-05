import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class CRF(nn.Module):

    def __init__(self, num_entities, pad_idx, bos_idx, eos_idx):
        super().__init__()
        self.num_entities = num_entities
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        

        self.transitions = nn.Parameter(torch.empty(self.num_entities, self.num_entities))  # treated as Parameter

        self._init_transitions()

    def forward(self, emissions, entities, mask=None):
        
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float, device='cuda' if torch.cuda.is_available() else 'cpu').bool()

        score = self._score(emissions, entities, mask)  # (B,)
        partition = self._log_partition(emissions, mask)  # (B,)
        return -torch.mean(score - partition)  # (1,)

    def _init_transitions(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # Enforce contraints with a big negative number (from 'row' to 'column')
        # so exp(-10000) will tend to zero
        self.transitions.data[:, self.bos_idx] = -10000  # no transition to BOS
        self.transitions.data[self.eos_idx, :] = -10000  # no transition from eos (except to PAD)
        self.transitions.data[self.pad_idx, :] = -10000  # no transition from pad (except to PAD)
        self.transitions.data[:, self.pad_idx] = -10000  # no transition to pad (except from PAD)
        self.transitions.data[self.pad_idx, self.pad_idx] = 0
        self.transitions.data[self.pad_idx, self.eos_idx] = 0

    def _score(self, emissions, entities, mask):
        
        batch_size, seq_len = entities.shape

        scores = torch.zeros(batch_size, dtype=torch.float, device='cuda' if torch.cuda.is_available() else 'cpu')

        first_ents = entities[:, 0]  # first entities (B,)
        last_valid_idx = mask.sum(1) - 1  # (B,)
        last_ents = entities.gather(1, last_valid_idx.unsqueeze(1)).squeeze()  # last entities (B,)

        t_scores = self.transitions[self.bos_idx, first_ents]

        e_scores = emissions[:, 0].gather(1, first_ents.unsqueeze(1)).squeeze()  # (B,)

        scores += e_scores + t_scores

        # For each remaining entities(words)
        for i in range(1, seq_len):
            prev_ents = entities[:, i - 1]  # previous entities
            curr_ents = entities[:, i]  # current entities

            # Calculate emission and transition scores
            e_scores = emissions[:, i].gather(1, curr_ents.unsqueeze(1)).squeeze()  # (B,)
            t_scores = self.transitions[prev_ents, curr_ents]  # (B,)

            # Apply masking --- If position is PAD, then the contributing score should be 0
            e_scores = e_scores * mask[:, i]
            t_scores = t_scores * mask[:, i]

            scores += e_scores + t_scores

        # Transition score from last entity to EOS
        scores += self.transitions[last_ents, self.eos_idx]

        return scores  # (B,)

    def _log_partition(self, emissions, mask):
        
        batch_size, seq_len, num_ents = emissions.shape

        # Antilog of partition part of score, which will be finally applied a logsumexp()
        alphas = self.transitions[self.bos_idx, :].unsqueeze(0) + emissions[:, 0]  # (B, num_ents)

        for i in range(1, seq_len):
            alpha_t = []

            for ent in range(num_ents):
                # Emission scores for current entity, broadcast to entity-wise
                e_scores = emissions[:, i, ent].unsqueeze(1)  # (B, 1)

                # Transition scores for current entity, broadcast to batch-wise
                t_scores = self.transitions[:, ent].unsqueeze(0)  # (1, num_entities)

                # Combine current scores with previous alphas (in log space: add instead of multiply)
                scores = e_scores + t_scores + alphas  # (B, num_entities)

                alpha_t.append(torch.logsumexp(scores, dim=1))  # inner (B,)

            new_alphas = torch.stack(alpha_t, dim=0).t()  # (B, num_entities)

            masking = mask[:, i].int().unsqueeze(1)  # (B, 1)
            alphas = masking * new_alphas + (1 - masking) * alphas

        # Transition scores from last entity to EOS
        end_scores = alphas + self.transitions[:, self.eos_idx].unsqueeze(0)  # (B, num_entities)

        return torch.logsumexp(end_scores, dim=1)

    def viterbi_decode(self, emissions, mask):
       
        batch_size, seq_len, num_ents = emissions.shape

        alphas = self.transitions[self.bos_idx, :].unsqueeze(0) + emissions[:, 0]  # (B, num_ents)

        backpointers = []

        for i in range(1, seq_len):
            alpha_t = []
            backpointers_t = []

            for ent in range(num_ents):
                # Emission scores for current entity, broadcast to entity-wise
                e_scores = emissions[:, i, ent].unsqueeze(1)  # (B, 1)

                # Transition scores for current entity, broadcast to batch-wise
                t_scores = self.transitions[:, ent].unsqueeze(0)  # (1, num_entities)

                # Combine current scores with previous alphas (in log space: add instead of multiply)
                scores = e_scores + t_scores + alphas  # (B, num_entities)

                # Find the highest score and the entity associated with it
                max_scores, max_ents = torch.max(scores, dim=-1)

                # Add the max score for current entity
                alpha_t.append(max_scores)  # inner: (B,)

                # Add the corresponding entity to backpointers
                backpointers_t.append(max_ents)  # backpointers_t finally (num_entities, B)

            new_alphas = torch.stack(alpha_t, dim=0).t()  # (B, num_entities)

            masking = mask[:, i].int().unsqueeze(1)  # (B, 1)
            alphas = masking * new_alphas + (1 - masking) * alphas

            backpointers.append(backpointers_t)  # finally (T-1, num_entities, B)

        # Transition scores from last entity to EOS
        end_scores = alphas + self.transitions[:, self.eos_idx].unsqueeze(0)  # (B, num_entities)

        # Final most probable score and corresponding entity
        max_final_scores, max_final_ents = torch.max(end_scores, dim=-1)  # (B,)

        # Decode the best sequence for current batch
        # Follow the backpointers to find the best sequence of entities
        best_seqs = []
        emission_lens = mask.sum(dim=1)  # (B,)

        for i in range(batch_size):
            sample_len = emission_lens[i].item()

            sample_final_entity = max_final_ents[i].item()

            sample_backpointers = backpointers[: sample_len - 1]

            best_path = [sample_final_entity]

            best_entity = sample_final_entity

            for backpointers_t in reversed(sample_backpointers):
                best_entity = backpointers_t[best_entity][i].item()
                best_path.insert(0, best_entity)

            best_seqs.append(best_path)  # best_seqs finally (B, *)

        return max_final_scores, best_seqs