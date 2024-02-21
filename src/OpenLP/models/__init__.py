from .modeling_bert import BertForLinkPredict
from .modeling_t5 import T5ForLinkPredict
from .modeling_bert_moe import BertMOEForLinkPredict
from .peft_models import BertPEFTForLinkPredict
from .bert_downstream import BertForDownstream

AutoModels = {
    'bert': BertForLinkPredict,
    'bert_moe': BertMOEForLinkPredict,
    't5': T5ForLinkPredict,
    'bert_peft': BertPEFTForLinkPredict,
    'bert_downstream': BertForDownstream
}
