#!/usr/bin/env python3
"""Generate semantic embeddings for hetero_test dataset.

Requires: pip install sentence-transformers numpy
Produces: hetero_features.npy (35 nodes x 384 dimensions, float32)

Node layout matches hetero_test.gql:
  0-19:  Paper nodes  (ML=0-6, DB=7-13, NLP=14-19)
  20-29: Author nodes
  30-34: Venue nodes
"""

import numpy as np
from sentence_transformers import SentenceTransformer

descriptions = [
    # Papers: ML (0-6)
    "Deep learning with convolutional neural networks for image classification",
    "Gradient descent optimization in large-scale neural network training",
    "Attention mechanisms and transformer architectures for sequence modeling",
    "Reinforcement learning with policy gradient methods in game environments",
    "Graph neural networks for node classification on citation networks",
    "Generative adversarial networks for synthetic image generation",
    "Transfer learning and fine-tuning pre-trained models for downstream tasks",
    # Papers: DB (7-13)
    "Query optimization in distributed relational database systems",
    "B-tree indexing structures for efficient range query processing",
    "Transaction isolation levels and concurrency control in OLTP workloads",
    "Graph database engines with worst-case optimal join algorithms",
    "Column-store compression techniques for analytical query processing",
    "Distributed consensus protocols for fault-tolerant data replication",
    "Approximate nearest neighbor search using locality-sensitive hashing",
    # Papers: NLP (14-19)
    "Word embeddings and distributional semantics for text similarity",
    "Named entity recognition using bidirectional LSTM-CRF models",
    "Machine translation with encoder-decoder attention networks",
    "Sentiment analysis of social media text using fine-tuned transformers",
    "Question answering with retrieval-augmented language models",
    "Text summarization using abstractive sequence-to-sequence models",
    # Authors (20-29)
    "Research scientist specializing in ML area",
    "Research scientist specializing in ML area",
    "Research scientist specializing in DB area",
    "Research scientist specializing in DB area",
    "Research scientist specializing in NLP area",
    "Research scientist specializing in NLP area",
    "Research scientist specializing in ML and DB area",
    "Research scientist specializing in DB and NLP area",
    "Research scientist specializing in ML area",
    "Research scientist specializing in NLP area",
    # Venues (30-34)
    "Academic conference focused on machine learning and neural information processing",
    "Academic conference focused on database management and data systems",
    "Academic conference focused on natural language processing and computational linguistics",
    "Academic conference focused on machine learning theory and applications",
    "Academic conference focused on very large databases and distributed data processing",
]

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descriptions, show_progress_bar=True).astype(np.float32)
    out_path = "hetero_features.npy"
    np.save(out_path, embeddings)
    print(f"Saved {embeddings.shape} float32 embeddings to {out_path}")
