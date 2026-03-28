from datasets import load_dataset
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf

FINAL_LABELS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral']


EMOTION_MAPPING = {
    "happy": ["joy", "amusement", "optimism", "excitement", "love", "pride", "relief"],
    "sad": ["sadness", "grief", "remorse", "disappointment", "embarrassment"],
    "angry": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "surprise": ["surprise", "realization"],
    "neutral": ["neutral", "confusion", "curiosity"]
}

LABEL2ID = {label: idx for idx, label in enumerate(FINAL_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

REVERSE_MAP = {}
for k, v in EMOTION_MAPPING.items():
    for emo in v:
        REVERSE_MAP[emo] = k


from datasets import load_dataset

dataset = load_dataset("go_emotions")
print(dataset)

PRIORITY = [
    "anger", "fear", "sadness", "disgust",
    "surprise", "joy", "neutral"
]

# Extract the label converter function once before mapping
# Assuming the 'labels' feature structure is consistent across all splits.
label_int2str_mapper = dataset["train"].features["labels"].feature.int2str

def map_labels(example):
    # Convert integer labels from 'labels' column to their string representations
    original_label_names = [label_int2str_mapper(label_id) for label_id in example["labels"]]

    for emo in PRIORITY:
        if emo in original_label_names and emo in REVERSE_MAP:
            example["label"] = LABEL2ID[REVERSE_MAP[emo]]
            return example
    example["label"] = -1
    return example


dataset = dataset.map(map_labels)
dataset = dataset.filter(lambda x: x["label"] != -1)
# Select all available examples in the filtered train set, as 30000 was out of bounds
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(len(dataset["train"])))

from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_data(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=160
    )

tokenized_dataset = dataset.map(tokenize_data, batched=True)


from transformers import TFDistilBertForSequenceClassification

model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(FINAL_LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    from_pt=True
)


from tf_keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = model.prepare_tf_dataset(
    tokenized_dataset["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator,
)

tf_val_dataset = model.prepare_tf_dataset(
    tokenized_dataset["validation"],
    shuffle=False,
    batch_size=32,
    collate_fn=data_collator,
)

tf_test_dataset = model.prepare_tf_dataset(
    tokenized_dataset["test"],
    shuffle=False,
    batch_size=32,
    collate_fn=data_collator,
)


from tf_keras.callbacks import EarlyStopping
import tensorflow as tf

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    tf_train_dataset,
    validation_data=tf_val_dataset,
    epochs=5,
    callbacks=[early_stop]
)


loss, accuracy = model.evaluate(tf_train_dataset)
print(f"Train Accuracy: {accuracy*100:.2f}%")

loss, accuracy = model.evaluate(tf_val_dataset)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

loss, accuracy = model.evaluate(tf_test_dataset)
print(f"Test Accuracy: {accuracy*100:.2f}%")


model.save_pretrained("/content/drive/MyDrive/emotion_model")
tokenizer.save_pretrained("/content/drive/MyDrive/emotion_model")