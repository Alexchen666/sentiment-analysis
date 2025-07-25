import re

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Set a maximum sequence length for padding/truncation
MAX_SEQ_LENGTH = 512
# Set a batch size for DataLoader
BATCH_SIZE = 32
# Training Hyperparameters
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5  # Common learning rate for Transformers
D_MODEL = 256  # Smaller for faster demo, typically 512 or 768
NUM_HEADS = 8  # Must divide embed_dim
FF_DIM = D_MODEL * 4  # Standard practice
NUM_LAYERS = 3  # Number of encoder layers
DROPOUT_RATE = 0.1
NUM_CLASSES = 2  # Positive/Negative sentiment


# Preprocessing
# Text cleaning (remove HTML tags, special characters)
def remove_html_tags(text):
    """Remove HTML tags from text."""
    clean = re.compile(
        "<.*?>"
    )  # Regex to match HTML tags, ? indicates non-greedy matching
    return re.sub(clean, "", text)


def remove_special_characters(text):
    """Remove special characters from text."""
    return re.sub(
        r"[^a-zA-Z0-9\s.,!?\"']", " ", text
    ).lower()  # Keep space and common punctuation marks


def clean_text(text):
    """Clean text by removing HTML tags and special characters."""
    text = remove_html_tags(text)
    text = remove_special_characters(text)
    return text


# Dataset Class
class IMDBDataset(Dataset):
    """
    Custom PyTorch Dataset for IMDB movie reviews.
    Handles text cleaning and tokenization using a Hugging Face tokenizer.
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Apply cleaning
        cleaned_text = clean_text(text)

        # Tokenize and encode using Hugging Face tokenizer
        # This handles tokenization, numericalization, padding, truncation,
        # and attention mask creation.
        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.max_len,  # Max length for padding/truncation
            padding="max_length",  # Pad to max_len
            truncation=True,  # Truncate if longer than max_len
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Squeeze to remove the batch dimension added by return_tensors='pt'
        # as __getitem__ expects to return single samples.
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Transformer
# --- 1. MultiHeadSelfAttention Module ---
# This module implements the self-attention mechanism, allowing the model
# to weigh the importance of different words in the input sequence.


class MultiHeadSelfAttention(nn.Module):
    """
    Implements the Multi-Head Self-Attention mechanism.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        # Ensure that d_model is divisible by num_heads
        assert d_model % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers for Query, Key, Value projections
        # These project the input into different spaces for each head.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output linear layer to combine the outputs of all heads
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for Multi-Head Self-Attention.

        Args:
            query (torch.Tensor): Input tensor for queries (batch_size, seq_len, d_model).
            key (torch.Tensor): Input tensor for keys (batch_size, seq_len, d_model).
            value (torch.Tensor): Input tensor for values (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): An optional mask tensor (batch_size, 1, 1, seq_len)
                                          to prevent attention to padded tokens.
                                          Typically 0 for padded positions, 1 for actual tokens.

        Returns:
            torch.Tensor: Output tensor after attention (batch_size, seq_len, d_model).
            torch.Tensor: Attention weights (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size = query.shape[0]

        # 1. Linear projections for Q, K, V
        # Shape after projection: (batch_size, seq_len, d_model)
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 2. Split into multiple heads and reshape
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        # Then permute to (batch_size, num_heads, seq_len, head_dim) for batch matrix multiplication
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Calculate attention scores (Q @ K_T)
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # 4. Apply mask (if provided)
        # Masking is typically used to ignore padding tokens.
        if mask is not None:
            # Expand mask to match attention_scores dimensions
            # mask shape: (batch_size, 1, 1, seq_len) -> (batch_size, 1, seq_len, seq_len)
            # The mask should be broadcastable.
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        # 5. Apply softmax to get attention probabilities
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 6. Apply dropout to attention weights
        attention_weights = self.attn_dropout(attention_weights)

        # 7. Multiply attention weights with V
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        context_layer = torch.matmul(attention_weights, V)

        # 8. Concatenate heads and reshape back to original embed_dim
        # Permute back to (batch_size, seq_len, num_heads, head_dim)
        # Then reshape to (batch_size, seq_len, embed_dim)
        # The permute() operation can make a tensor non-contiguous.
        # Since the subsequent view() operation requires a contiguous tensor to reshape
        # .contiguous() is called in between to ensure the memory layout is correct for the view() operation to succeed.
        context_layer = (
            context_layer.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )

        # 9. Final linear projection
        output = self.out_proj(context_layer)

        return output, attention_weights


# --- 2. PositionalEncoding Module ---
# Transformers are permutation-invariant, meaning they don't inherently understand
# the order of words. Positional Encoding adds information about the position
# of each token in the sequence.


class PositionalEncoding(nn.Module):
    """
    Implements the Positional Encoding mechanism.
    Adds sinusoidal positional encodings to the input embeddings.

    Args:
        d_model (int): The dimension of the input embeddings.
        max_seq_len (int): The maximum sequence length the model is expected to handle.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix
        # pe shape: (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # position shape: (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # div_term shape: (d_moel / 2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # Apply sine to even indices in pe, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for batch (1, max_seq_len, d_model)
        # This allows it to be broadcasted to input_embeddings (batch_size, seq_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))  # 'pe' is not a learnable parameter

    def forward(self, x):
        """
        Forward pass for Positional Encoding.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encodings added.
        """
        # Add positional encoding to the input embeddings
        # x is (batch_size, seq_len, d_model)
        # self.pe is (1, max_seq_len, d_model)
        # We slice self.pe to match the current sequence length of x
        x = x + self.pe[:, : x.size(1), :]  # type: ignore
        return self.dropout(x)  # Apply dropout to the output


# --- 3. TransformerEncoderLayer Module ---
# This is a single layer of the Transformer Encoder, consisting of
# Multi-Head Self-Attention, a Feed-Forward Network, and Layer Normalization.


class TransformerEncoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer Encoder.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        ff_dim (int): The dimension of the feed-forward network's hidden layer.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass for a single Transformer Encoder Layer.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Attention mask (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor from this encoder layer.
        """
        attn_output, _ = self.self_attn(
            x, x, x, mask
        )  # Q, K, V are all from x for self-attention
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization

        ff_output = self.ff(x)  # Feed-forward network
        x = x + self.dropout2(ff_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization

        return x


# --- 4. TransformerEncoder Module ---
# This stacks multiple TransformerEncoderLayer instances to form the full encoder.


class TransformerEncoder(nn.Module):
    """
    Implements the full Transformer Encoder, stacking multiple Encoder Layers.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        ff_dim (int): The dimension of the feed-forward network's hidden layer.
        num_layers (int): The number of TransformerEncoderLayer instances to stack.
        max_seq_len (int): The maximum sequence length for positional encoding.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        ff_dim,
        num_layers,
        max_seq_len,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Stack multiple TransformerEncoderLayer instances
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, src_mask=None):
        """
        Forward pass for the Transformer Encoder.

        Args:
            src (torch.Tensor): Input tensor of token IDs (batch_size, seq_len).
            src_mask (torch.Tensor, optional): Source attention mask (batch_size, 1, 1, seq_len).
                                              This mask should be 0 for padded tokens.

        Returns:
            torch.Tensor: Output tensor from the final encoder layer (batch_size, seq_len, d_model).
        """
        # Convert token IDs to embeddings
        x = self.embedding(src)
        # Add positional encoding
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


# Sentiment Classification Model
class SentimentTransformer(nn.Module):
    """
    A Transformer-based model for binary sentiment classification.
    Consists of a TransformerEncoder followed by a classification head.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        ff_dim,
        num_layers,
        max_seq_len,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.classifier = nn.Linear(d_model, num_classes)  # Final classification layer

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the SentimentTransformer.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs (batch_size, seq_len).
            attention_mask (torch.Tensor): Tensor indicating actual tokens (1) and padding (0)
                                          (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits for each class (batch_size, num_classes).
        """
        # Create the mask for the TransformerEncoder.
        # The mask needs to be (batch_size, 1, 1, seq_len) for MultiHeadSelfAttention.
        # It's typically 0 for padded positions and 1 for actual tokens.
        # We convert the attention_mask from (batch_size, seq_len) to (batch_size, 1, 1, seq_len)
        # and ensure it's a boolean mask for masked_fill.
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()

        # Output shape: (batch_size, seq_len, embed_dim)
        encoder_output = self.transformer_encoder(input_ids, src_mask)

        # For classification, we typically take the output corresponding to the [CLS] token.
        # The [CLS] token is usually the first token in the sequence (index 0).
        cls_token_output = encoder_output[:, 0, :]  # Shape: (batch_size, embed_dim)

        # Pass the [CLS] token output through the classification head
        logits = self.classifier(cls_token_output)  # Shape: (batch_size, num_classes)

        return logits


def train_model(model, data_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()  # Zero the gradients

        outputs = model(input_ids, attention_mask)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluates the model on the given data.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def main():
    df = pl.read_csv("data/imdb.csv")

    # Replace sentiment values with integers
    # 1 for positive, 0 for negative
    df = df.with_columns(
        pl.col("sentiment").replace("positive", 1).replace("negative", 0).cast(pl.Int8)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    X_train = X_train.to_list()
    X_test = X_test.to_list()
    y_train = y_train.to_list()
    y_test = y_test.to_list()

    # Tokenisation
    # The tokenizer from HuggingFace handles vocabulary building, sequence padding, and truncation as well.
    # Load a pre-trained tokenizer. 'bert-base-uncased' is a good general-purpose model.
    # The 'uncased' means it expects lowercase input, which aligns with our cleaning.
    # Setting `do_lower_case=False` because we already lowercased the text.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)

    # Create Dataset instances
    train_dataset = IMDBDataset(X_train, y_train, tokenizer, MAX_SEQ_LENGTH)
    test_dataset = IMDBDataset(X_test, y_test, tokenizer, MAX_SEQ_LENGTH)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Device configuration (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of train batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print("-" * 30)

    print("\n--- Initializing Model, Loss, and Optimizer ---")
    # Initialize the Sentiment Transformer model
    model = SentimentTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LENGTH,
        dropout=DROPOUT_RATE,
        num_classes=NUM_CLASSES,
    ).to(device)  # Move model to the selected device

    # Define Loss Function (CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()

    # Define Optimizer (AdamW is common for Transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )
    print("-" * 30)

    print("\n--- Starting Training ---")
    test_acc = 0.0  # Initialize test accuracy for final report
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_model(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"Epoch {epoch}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print("-" * 30)

    print("\n--- Training Complete! ---")
    print(f"Final Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
