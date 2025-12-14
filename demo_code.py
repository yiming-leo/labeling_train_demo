import pandas as pd
from datasets import load_dataset
from openai import OpenAI


def label_them(api_key):
    # standard AG News convention
    label_desc = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    # import ag_news for labeling different types' topic with 100 records
    dataset = load_dataset("kenhktsui/test_ag_news")

    train_data = dataset["train"].select(range(30))
    # test_data = dataset["test"]

    # https: // bailian.console.aliyun.com / & tab = doc?tab = api  # /api/?type=model&url=2712576
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    results = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for record in train_data:
        text = record["text"]
        human_label = label_desc[record["label"]]

        # ref: https://help.aliyun.com/zh/model-studio/getting-started/models
        completion = client.chat.completions.create(
            # select specific models
            model="qwen-plus",
            # the content want to pass
            messages=[
                # AI plays as a helpful assistant's role
                {
                    "role": "system",
                    "content": (
                        "You are a professional news editor.\n"
                        "Your task is to classify news articles into one of the following categories:\n"
                        "World, Sports, Business, Sci/Tech.\n"
                        "Return only one category name."
                    )
                },
                # user's question
                {
                    "role": "user",
                    "content": f"News Articles:\n{text}"
                }
            ],
            # [0,2)
            temperature=0
        )

        # choose response's useful content, and - \n
        ai_label = completion.choices[0].message.content.strip()

        # calculate token cost
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens
        total_tokens += completion.usage.total_tokens

        # append to a record, for analyzing
        results.append(
            {
                "text": text,
                "human_label": human_label,
                "ai_label": ai_label
            }
        )

    print(f"cost: prompt_token: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}")

    # save the results
    df = pd.DataFrame(results)
    df.to_csv("labeled.csv", index=False)
    print("labeled.csv saved")
