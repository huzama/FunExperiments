import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torch_xla.runtime as xr
import wandb
from datasets import load_dataset
from torch import nn
from torch_xla.experimental.xla_sharding import Mesh
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

# Enable XLA SPMD execution mode.
xr.use_spmd()

# Initalize wandb
wandb.init(project="test", entity="huzama")


# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()
mesh_shape = (2, num_devices // 2)  # 2x4 on v3-8, 2x2 on v4-8
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ("model", "data"))

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16
).to(xm.xla_device())

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

eli5 = load_dataset("eli5", split="train_asks[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers"]])


tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    remove_columns=eli5["train"].column_names,
)

block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized_eli5.map(group_texts, batched=True)


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Dataloader

train_dataloader = torch.utils.data.DataLoader(
    lm_dataset["train"],
    batch_size=2,
    collate_fn=data_collator,
    shuffle=True,
    drop_last=True,
)
# Training loop

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
partition_spec = ("model", "data")

for i in model.parameters():
    if i.requires_grad and len(i.shape) == 2:
        xs.mark_sharding(i, mesh, partition_spec)


model.train()
for i, data in tqdm(enumerate(train_dataloader)):
    # Assumes `loader` returns data, target on XLA device
    optimizer.zero_grad()
    # Sharding annotate input data, we can shard any input
    # dimensions. Sharding the batch dimension enables
    # data parallelism, sharding the feature dimension enables
    # spatial partitioning.

    inputs = data["input_ids"]
    labels = data["labels"]

    inputs = inputs.to(xm.xla_device())
    labels = labels.to(xm.xla_device())

    xs.mark_sharding(inputs, mesh, partition_spec)
    xs.mark_sharding(labels, mesh, partition_spec)

    ouput = model(inputs)["logits"]
    B, S, E = ouput.shape
    loss = loss_fn(ouput.reshape(B * S, -1), inputs.reshape(B * S))
    loss.backward()
    wandb.log({"loss": loss.item()})
    optimizer.step()
    xm.mark_step()
