import bitagent
import transformers
from bitagent.miners.context_util import get_relevant_context_and_citations_from_synapse
import bittensor as bt
from bitagent.Miner_Response import Miner_Model

def miner_init(self, config=None):
    transformers.logging.set_verbosity_error()

def llm(input_text):
    result = Miner_Model(input_text)
    return result

def miner_process(self, synapse: bitagent.protocol.QnATask) -> bitagent.protocol.QnATask:
    if not synapse.urls and not synapse.datas:
        context = ""
        citations = []
    else:
        context, citations = get_relevant_context_and_citations_from_synapse(synapse)
    query_text = f"Please provide the user with an answer to their question: {synapse.prompt}.\n\n Response: "
    if context:
        query_text = f"Given the following CONTEXT:\n\n{context}\n\n{query_text}"
        
    llm_response = self.llm(query_text)
    synapse.response["response"] = llm_response
    synapse.response["citations"] = citations
    
    
    return synapse
