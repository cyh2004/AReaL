from datasets import load_dataset


def get_parquet_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset("parquet", data_files=path)["train"]

    def process(sample):
        messages = sample["prompt"]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["prompt"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset