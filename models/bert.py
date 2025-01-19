import torch
import pytorch_pretrained_bert as bert

class BERT_MODEL:
    def __init__(self, version='bert-base-cased'):
        self.tokenizer = bert.BertTokenizer.from_pretrained(version)
        self.model = bert.BertModel.from_pretrained(version, cache_dir = "bert-base-cache")
        self.model.eval()


    def encode(self, texts):
        ''' Use tokenizer and model to encode texts '''
        encs = {}
        for text in texts:
            tokenized = self.tokenizer.tokenize(text)
            indexed = self.tokenizer.convert_tokens_to_ids(tokenized)
            segment_idxs = [0] * len(tokenized)
            tokens_tensor = torch.tensor([indexed])
            segments_tensor = torch.tensor([segment_idxs])
            enc, _ = self.model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)
            enc = enc[:, 0, :]  
            encs[text] = enc.detach().view(-1).numpy()
        return encs
