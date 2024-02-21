from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding, DefaultDataCollator, PreTrainedTokenizer, DataCollatorForLanguageModeling

from IPython import embed

@dataclass
class TrainCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]
        task_ids = [f["task_id"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        task_ids = torch.LongTensor(task_ids)

        # return {'center_input': q_collated, 'task_ids': task_ids}, \
        #          {'center_input': k_collated, 'task_ids': task_ids}
        return {**q_collated, 'task_ids': task_ids}, \
                 {**k_collated, 'task_ids': task_ids}


@dataclass
class TrainMatchCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {**q_collated}, {**k_collated}


@dataclass
class TrainRerankCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]
        q_n = [f["query_n"] for f in features]
        k_n = [f["key_n"] for f in features]
        q_mask = [f["query_n_mask"] for f in features]
        k_mask = [f["key_n_mask"] for f in features]
        label_mask = [f["label_mask"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])
        if isinstance(q_n[0], list):
            q_n = sum(q_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])
        if isinstance(k_n[0], list):
            k_n = sum(k_n, [])
        if isinstance(k_mask[0], list):
            k_mask = sum(k_mask, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        qn_collated = self.tokenizer.pad(
            q_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        kn_collated = self.tokenizer.pad(
            k_n,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        q_mask = torch.LongTensor(q_mask)
        k_mask = torch.LongTensor(k_mask)
        label_mask = torch.LongTensor(label_mask)

        return {'center_input': q_collated, 'neighbor_input': qn_collated, 'mask': q_mask}, \
                 {'center_input': k_collated, 'neighbor_input': kn_collated, 'mask': k_mask, 'label_mask':label_mask}


@dataclass
class TrainNCCCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        labels = [f["label"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = torch.LongTensor(labels)

        return {**q_collated}, labels


@dataclass
class TrainREGCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        labels = [f["label"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = torch.FloatTensor(labels)

        return {**q_collated}, labels


@dataclass
class TrainHnCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 32

    def __call__(self, features):

        qq = [f["query"] for f in features]
        kk = [f["key"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(kk[0], list):
            kk = sum(kk, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        k_collated = self.tokenizer.pad(
            kk,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {**q_collated}, {**k_collated}


@dataclass
class EncodeCollator(DefaultDataCollator):
    def __call__(self, features):

        text_ids = [x["text_id"] for x in features]
        input_ids = [x["input_ids"] for x in features]
        attention_mask = [x["attention_mask"] for x in features]
        token_type_ids = [x["token_type_ids"] for x in features]

        return text_ids, {'input_ids': torch.LongTensor(input_ids), 'attention_mask': torch.LongTensor(attention_mask), 'token_type_ids': torch.LongTensor(token_type_ids)}
