from transformers import pipeline


def task_text_classification():#定义一个函数，叫做“任务_文本分类”。
    """使用默认的小型情感分析模型进行分类"""
    print("--- 任务 1.1: 文本分类 (基础版) ---")

    print("正在加载情感分析模型...")
    # 修改后：添加 framework='pt' 来指定使用 PyTorch 框架
    classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english" ,framework='pt')
    print("模型加载完成！")

    text_list = [
        "This is such a beautiful day!",
        "I am not happy with the service.",

        "The movie was okay, not great but not bad either.",


    ]

    print("正在进行预测...")
    results = classifier(text_list)

    for text, result in zip(text_list, results):
        print(f"文本: '{text}'")
        print(f"  -> 标签: {result['label']}, 分数: {result['score']:.4f}")


def task_text_generation():
    """使用经典的 GPT-2 模型进行文本生成"""
    print("\n--- 任务 1.2: 文本生成 (基础版) ---")

    print("正在加载 GPT-2 模型...")
    # 修改后：同样为 GPT-2 指定框架，保持代码一致性
    generator = pipeline('text-generation', model='gpt2', framework='pt')
    print("模型加载完成！")

    prompt = "In a world where AI is taking over,"

    print(f"正在根据 '{prompt}' 生成文本...")
    # max_new_tokens 只控制新生成的文本长度，更精确
    generated_texts = generator(prompt, max_new_tokens=30, num_return_sequences=1)

    print("-" * 20)
    print(generated_texts[0]['generated_text'])


if __name__ == "__main__":
    task_text_classification()
    task_text_generation()