import base64
from openai import OpenAI

ans_hist = []
client = OpenAI(
    base_url="https://ai-gateway.vei.volces.com/v1",
    api_key="sk-63c9bcbf8cff49c1a0042aece4097685d92zjbjbo65jwcux",
)
cnt = 0

def llm_infer(text):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Path to your image
    image_path = "/root/rdk_model_zoo/demos/captured_images/1.jpg"
    # image_path = "/home/ros/share_dir/gitrepos/volcano-engine/imgs/2.jpg"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    # Build the conversation history for the messages list
    messages = [{"role": "system", "content": "你现在是一个负责给机器人生成规划代码的模块。"},]
    
    # Include previous conversation history if available
    for entry in ans_hist:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["assistant"]})

    # Append the current user message and image content
    global cnt
    if cnt == 0:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ]
        })
    cnt += 1
    # print(messages)

    # Make the API call with the updated message history
    completion = client.chat.completions.create(
        model="qwen-vl-8k",
        messages=messages,
    )

    # Get the response message and store both user and assistant responses
    response_message = completion.choices[0].message.content
    ans_hist.append({
        "user": text,
        "assistant": response_message
    })

    # Print the assistant's response
    print("Answer:", response_message)

# Example usage

# while True:
#     text = input()
#     llm_infer(text)
