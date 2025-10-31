import torch.nn as nn
import torch
import os


class BERTSeqClf(nn.Module):
    def __init__(self, num_labels, num_target_labels, n_layers_freeze=0, n_layers_freeze_wiki=0):
        super(BERTSeqClf, self).__init__()

        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoModel

        self.bert_shared = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_wiki = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_google = AutoModel.from_pretrained('bert-base-uncased')

        # model_dir = "../../autodl-tmp/bertweet"

        self.bert_tweet = AutoModel.from_pretrained('vinai/bertweet-base')

        # self.bert_tweet = AutoModel.from_pretrained('vinai/bertweet-base')

        config = self.bert_shared.config
        config_wiki = self.bert_wiki.config
        config_tweet = self.bert_tweet.config
        config_google = self.bert_google.config

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_wiki = nn.Dropout(config_wiki.hidden_dropout_prob)
        self.dropout_tweet = nn.Dropout(config_tweet.hidden_dropout_prob)
        self.dropout_google = nn.Dropout(config_google.hidden_dropout_prob)

        for i in range(20):
            exec(
                'self.encoder{} = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=12, dropout=0.1, batch_first=True)'.format(
                    i))

        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.feed = nn.Linear(num_labels, 1)
        self.target_classifier = nn.Linear(config_wiki.hidden_size, num_target_labels)
        self.auxiliary_classifier = nn.Linear(config_tweet.hidden_size, num_labels)

    def forward(self, input_ids_shared=None, attention_mask_shared=None, token_type_ids_shared=None,
                input_ids_wiki=None, attention_mask_wiki=None, input_ids_google=None, attention_mask_google=None,
                input_ids_tweet=None, attention_mask_tweet=None, token_type_ids_tweet=None):
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

        # print("pooled_output_google", pooled_output_google.shape)

        outputs_tweet = self.bert_tweet(input_ids=input_ids_tweet,
                                        attention_mask=attention_mask_tweet,
                                        token_type_ids=token_type_ids_tweet,
                                        return_dict=True)

        pooled_output_tweet = outputs_tweet.pooler_output
        pooled_output_tweet = self.dropout_tweet(pooled_output_tweet)
        pooled_output_tweet = torch.unsqueeze(pooled_output_tweet, 1)

        local_shared_1 = self.encoder17(self.encoder16(torch.concat([pooled_output_google, pooled_output_wiki], dim=1)))
        # print("local_shared_1", local_shared_1.shape)

        first1 = self.encoder13(self.encoder12(torch.concat([pooled_output_google, local_shared_1], dim=1)))

        left1 = self.encoder1(self.encoder0(torch.concat([pooled_output_wiki, local_shared_1], dim=1)))

        medium1 = self.encoder3(self.encoder2(
            torch.concat([pooled_output_google, pooled_output_wiki, pooled_ouptut_shared, pooled_output_tweet], dim=1)))

        medium_individual_1 = self.encoder3(self.encoder2(pooled_ouptut_shared))

        right1 = self.encoder5(self.encoder4(torch.concat([pooled_ouptut_shared, pooled_output_tweet], dim=1)))

        local_shared_2 = self.encoder19(self.encoder18(torch.concat([first1, left1], dim=1)))

        first2 = self.encoder15(self.encoder14(torch.concat([first1, local_shared_2], dim=1)))

        left2 = self.encoder7(self.encoder6(torch.concat([left1, local_shared_2], dim=1)))

        medium2 = self.encoder9(self.encoder8(torch.concat([first1, left1, medium1, right1], dim=1)))

        medium_individual_2 = self.encoder9(self.encoder8(medium_individual_1))

        right2 = self.encoder11(self.encoder10(torch.concat([medium1, right1], dim=1)))

        medium2 = torch.mean(medium2, 1)

        right2 = torch.mean(right2, 1)

        first2 = torch.mean(first2, 1)

        left2 = torch.mean(left2, 1)

        logits_first = self.target_classifier(torch.squeeze(first2, 1))

        logits_target = self.target_classifier(torch.squeeze(left2, 1))

        logits = self.classifier(medium2)

        y_pred = torch.sigmoid(self.feed(logits))

        logits_individual = self.classifier(torch.squeeze(medium_individual_2, 1))

        y_individual = torch.sigmoid(self.feed(logits_individual))

        logits_auxiliary = self.auxiliary_classifier(right2)

        # return logits_target, logits, logits_auxiliary

        # return logits_target, logits, y_pred, logits_individual, y_individual, logits_auxiliary

        return logits_first, logits_target, logits, y_pred, logits_individual, y_individual, logits_auxiliary