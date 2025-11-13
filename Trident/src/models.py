import torch.nn as nn
import torch
import torch.nn.functional as F
import os


class ExpertNetwork(nn.Module):
    """Single expert in MoE"""

    def __init__(self, hidden_size, expert_hidden_size=None):
        super(ExpertNetwork, self).__init__()
        if expert_hidden_size is None:
            expert_hidden_size = hidden_size * 4

        self.fc1 = nn.Linear(hidden_size, expert_hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(expert_hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.layer_norm(out + identity)
        return out


class MoELayer(nn.Module):
    """Mixture of Experts with Top-K gating"""

    def __init__(self, hidden_size, num_experts=8, top_k=2, expert_hidden_size=None):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        # Create expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(hidden_size, expert_hidden_size)
            for _ in range(num_experts)
        ])

        # Gating network with layer norm
        self.gate_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
        original_shape = x.shape
        needs_reshape = len(original_shape) == 3

        if needs_reshape:
            batch_size, seq_len, hidden_size = x.shape
            x_flat = x.reshape(-1, hidden_size)
        else:
            x_flat = x
            batch_size = x.shape[0]

        # Compute gating scores with normalization
        gate_input = self.gate_norm(x_flat)
        gate_logits = self.gate(gate_input)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-K gating
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute outputs from all experts
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_flat)
            outputs.append(expert_out)

        expert_outputs = torch.stack(outputs, dim=1)  # [batch*seq, num_experts, hidden]

        # Gather top-k expert outputs
        num_tokens = x_flat.shape[0]
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, self.top_k)
        selected_outputs = expert_outputs[token_indices, top_k_indices]  # [batch*seq, top_k, hidden]

        # Weighted combination
        final_output = torch.sum(selected_outputs * top_k_probs.unsqueeze(-1), dim=1)

        if needs_reshape:
            final_output = final_output.reshape(batch_size, seq_len, hidden_size)

        return final_output


class BERTSeqClf(nn.Module):
    def __init__(self, num_labels, num_target_labels, n_layers_freeze=0, n_layers_freeze_wiki=0):
        super(BERTSeqClf, self).__init__()

        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoModel

        self.bert_shared = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_wiki = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_google = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_tweet = AutoModel.from_pretrained('bert-base-uncased')

        config = self.bert_shared.config
        config_wiki = self.bert_wiki.config
        config_tweet = self.bert_tweet.config
        config_google = self.bert_google.config

        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_wiki = nn.Dropout(config_wiki.hidden_dropout_prob)
        self.dropout_tweet = nn.Dropout(config_tweet.hidden_dropout_prob)
        self.dropout_google = nn.Dropout(config_google.hidden_dropout_prob)

        # Original transformer encoder layers
        for i in range(20):
            exec(
                'self.encoder{} = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=12, dropout=0.1, batch_first=True)'.format(
                    i))

        # MoE Group 1: Knowledge Integration (Wiki + Google background knowledge)
        self.knowledge_moe_1 = MoELayer(hidden_size, num_experts=8, top_k=2)
        self.knowledge_moe_2 = MoELayer(hidden_size, num_experts=8, top_k=2)

        # MoE Group 2: Tweet Knowledge Integration
        self.tweet_moe_1 = MoELayer(hidden_size, num_experts=6, top_k=2)
        self.tweet_moe_2 = MoELayer(hidden_size, num_experts=6, top_k=2)

        # MoE Group 3: Multi-source Fusion (Shared branch)
        self.fusion_moe_1 = MoELayer(hidden_size, num_experts=10, top_k=2)
        self.fusion_moe_2 = MoELayer(hidden_size, num_experts=10, top_k=2)

        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.feed = nn.Linear(num_labels, 1)
        self.target_classifier = nn.Linear(config_wiki.hidden_size, num_target_labels)
        self.auxiliary_classifier = nn.Linear(config_tweet.hidden_size, num_labels)

    def forward(self, input_ids_shared=None, attention_mask_shared=None, token_type_ids_shared=None,
                input_ids_wiki=None, attention_mask_wiki=None, input_ids_google=None, attention_mask_google=None,
                input_ids_tweet=None, attention_mask_tweet=None, token_type_ids_tweet=None):
        # Encode all inputs
        outputs = self.bert_shared(input_ids=input_ids_shared,
                                   attention_mask=attention_mask_shared,
                                   token_type_ids=token_type_ids_shared,
                                   return_dict=True)

        pooled_ouptut_shared = outputs.pooler_output
        pooled_ouptut_shared = self.dropout(pooled_ouptut_shared)
        pooled_ouptut_shared = torch.unsqueeze(pooled_ouptut_shared, 1)

        outputs_wiki = self.bert_wiki(input_ids=input_ids_wiki,
                                      attention_mask=attention_mask_wiki,
                                      return_dict=True)

        pooled_output_wiki = outputs_wiki.pooler_output
        pooled_output_wiki = self.dropout_wiki(pooled_output_wiki)
        pooled_output_wiki = torch.unsqueeze(pooled_output_wiki, 1)

        outputs_google = self.bert_google(input_ids=input_ids_google,
                                          attention_mask=attention_mask_google,
                                          return_dict=True)

        pooled_output_google = outputs_google.pooler_output
        pooled_output_google = self.dropout_google(pooled_output_google)
        pooled_output_google = torch.unsqueeze(pooled_output_google, 1)

        outputs_tweet = self.bert_tweet(input_ids=input_ids_tweet,
                                        attention_mask=attention_mask_tweet,
                                        token_type_ids=token_type_ids_tweet,
                                        return_dict=True)

        pooled_output_tweet = outputs_tweet.pooler_output
        pooled_output_tweet = self.dropout_tweet(pooled_output_tweet)
        pooled_output_tweet = torch.unsqueeze(pooled_output_tweet, 1)

        # ===== MoE Group 1: Knowledge Integration (Google + Wiki) =====
        # First stage: integrate raw background knowledge
        knowledge_concat = torch.concat([pooled_output_google, pooled_output_wiki], dim=1)
        knowledge_moe_out = self.knowledge_moe_1(knowledge_concat)

        local_shared_1 = self.encoder17(self.encoder16(knowledge_moe_out))

        # Second stage: refine with MoE
        local_shared_1 = self.knowledge_moe_2(local_shared_1)

        first1 = self.encoder13(self.encoder12(torch.concat([pooled_output_google, local_shared_1], dim=1)))
        left1 = self.encoder1(self.encoder0(torch.concat([pooled_output_wiki, local_shared_1], dim=1)))

        # ===== MoE Group 2: Tweet Knowledge Integration =====
        # First stage: integrate tweet with context
        tweet_concat = torch.concat([pooled_ouptut_shared, pooled_output_tweet], dim=1)
        tweet_moe_out = self.tweet_moe_1(tweet_concat)

        right1 = self.encoder5(self.encoder4(tweet_moe_out))

        # Second stage: refine tweet knowledge
        right1 = self.tweet_moe_2(right1)

        # ===== MoE Group 3: Multi-source Fusion (Shared Branch) =====
        # First stage: integrate all knowledge sources
        fusion_concat = torch.concat([pooled_output_google, pooled_output_wiki,
                                      pooled_ouptut_shared, pooled_output_tweet], dim=1)
        fusion_moe_out = self.fusion_moe_1(fusion_concat)

        medium1 = self.encoder3(self.encoder2(fusion_moe_out))
        medium_individual_1 = self.encoder3(self.encoder2(pooled_ouptut_shared))

        # Second stage processing
        local_shared_2 = self.encoder19(self.encoder18(torch.concat([first1, left1], dim=1)))

        first2 = self.encoder15(self.encoder14(torch.concat([first1, local_shared_2], dim=1)))
        left2 = self.encoder7(self.encoder6(torch.concat([left1, local_shared_2], dim=1)))

        # Second stage fusion: refine integrated knowledge with MoE
        medium_fusion_concat = torch.concat([first1, left1, medium1, right1], dim=1)
        medium_fusion_out = self.fusion_moe_2(medium_fusion_concat)

        medium2 = self.encoder9(self.encoder8(medium_fusion_out))
        medium_individual_2 = self.encoder9(self.encoder8(medium_individual_1))

        right2 = self.encoder11(self.encoder10(torch.concat([medium1, right1], dim=1)))

        # Final pooling
        medium2 = torch.mean(medium2, 1)
        right2 = torch.mean(right2, 1)
        first2 = torch.mean(first2, 1)
        left2 = torch.mean(left2, 1)

        # Classification heads
        logits_first = self.target_classifier(torch.squeeze(first2, 1))
        logits_target = self.target_classifier(torch.squeeze(left2, 1))
        logits = self.classifier(medium2)
        y_pred = torch.sigmoid(self.feed(logits))
        logits_individual = self.classifier(torch.squeeze(medium_individual_2, 1))
        y_individual = torch.sigmoid(self.feed(logits_individual))
        logits_auxiliary = self.auxiliary_classifier(right2)

        return logits_first, logits_target, logits, y_pred, logits_individual, y_individual, logits_auxiliary