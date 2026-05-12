from __future__ import annotations


SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."

BASE_TEMPLATES = {
    "ruler_qa": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp}\n<User> The following context is the documents I have read: \n{context}\n <Assistant> I have learned the documents and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
            "rag_agent": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\n Now Answer the Question: {question}",
            "agentic_memory_agent": "Search Archival Memory and answer my question. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
        },
    },
    "longmemeval": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant \n<User> The following context is the conversation between the user and the assistant: \n{context}\n <Assistant> I have memorized the conversation and I will answer the question you ask.",
        "query": {
            "long_context_agent": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "rag_agent": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "agentic_memory_agent": "Search Archival Memory and answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
        },
    },
    "eventqa": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp}\n<User> The following context is the book excerpt: \n{context}\n <Assistant> I have read the book excerpt and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
            "rag_agent": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
            "agentic_memory_agent": "Search Archival Memory, complete the task below:\n\n{question}\n\n The event that happens next is:",
        },
    },
    "in_context_learning": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the examples I have learned: \n{context}\n <Assistant> I have learned the examples and I will answer the question you ask.",
        "query": {
            "long_context_agent": 'Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {label}" and nothing else. \n\n{question} \n\n label:',
            "rag_agent": 'Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {label}" and nothing else. \n\nQuestion:{question} \n\n label:',
            "agentic_memory_agent": 'Search Archival Memory and use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {label}" and nothing else. \n\n{question} \n\n label:',
        },
    },
    "recsys_redial": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the dialogues between a user and recommender system: \n{context}\n <Assistant> I have memorized the dialogues and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
            "rag_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
            "agentic_memory_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Search Archival Memory, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
        },
    },
    "infbench_sum": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
        "query": {
            "long_context_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
            "rag_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
            "agentic_memory_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
        },
    },
    "detective_qa": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
            "rag_agent": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
            "agentic_memory_agent": "Search Archival Memory and answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
        },
    },
    "factconsolidation": {
        "system": SYSTEM_MESSAGE,
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.",
        "query": {
            "long_context_agent": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
            "rag_agent": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
            "agentic_memory_agent": "Pretend you are a knowledge management system. Each fact in the  Archival Memory is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the Archival Memory by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Archival Memory] \n\n Question: Based on the Archival Memory, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the  Archival Memory, {question} \nAnswer:",
        },
    },
}

AGENT_TYPE_MAPPING = {
    "rag": "rag_agent",
    "Long_context_agent": "long_context_agent",
    "Agentic_memory": "agentic_memory_agent",
}

DATASET_MAPPING = {
    ("ruler_", "qa"): "ruler_qa",
    ("icl_",): "in_context_learning",
    ("infbench_", "sum"): "infbench_sum",
    ("eventqa_",): "eventqa",
    ("recsys_", "redial"): "recsys_redial",
    ("longmemeval_",): "longmemeval",
    ("factconsolidation_",): "factconsolidation",
    ("detective_", "qa"): "detective_qa",
}


def normalize_agent_name(agent_name: str) -> str:
    for pattern, normalized_name in AGENT_TYPE_MAPPING.items():
        if pattern in agent_name:
            return normalized_name
    raise NotImplementedError(f"Unknown agent type: {agent_name}")


def normalize_dataset_name(sub_dataset: str) -> str:
    for patterns, normalized_name in DATASET_MAPPING.items():
        if all(pattern in sub_dataset for pattern in patterns):
            return normalized_name
    raise NotImplementedError(f"Unknown dataset: {sub_dataset}")


def get_template(sub_dataset: str, template_name: str, agent_name: str) -> str:
    normalized_agent = normalize_agent_name(agent_name)
    normalized_dataset = normalize_dataset_name(sub_dataset)
    base_template = BASE_TEMPLATES[normalized_dataset][template_name]
    if isinstance(base_template, dict):
        return base_template[normalized_agent]
    return base_template
