import gradio as gr
import numpy as np
from PIL import Image
import time
import os

import fawkes.protection as fp

def process(input_image, level):
    # 获取当前时间
    now_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    input_img_name = f"./imgs/{now_time}_input.png"
    output_img_name = input_img_name.replace("input", f"output_{level}")
    # 保存图片
    img = Image.fromarray(np.uint8(input_image))
    img.save(input_img_name)

    # try:
    protector = fp.Fawkes("arcface_extractor_0", 0, 1, mode=level)

    protector.run_protection([input_img_name], th=0.01, sd=1e6, lr=2,
                            max_step=1000,
                            batch_size=1, format="png",
                            separate_target='store_true', debug='store_true', no_align='store_true')
    # except Exception as e:
    #     print(e)
    #     print("Error")
    #     return input_img_name

    # 重命名文件
    os.rename(input_img_name.replace("input", f"input_cloaked"), 
              output_img_name)

    return output_img_name

def main():
    with gr.Blocks() as demo:
        # 图片
        with gr.Row():
            # 上传图片
            input_image = gr.Image(height=500, source='upload')
            # 输出图片
            ouput_image = gr.Image(height=500)

        # 参数
        with gr.Row():
            # 伪造程度
            level_drop = gr.Dropdown(["low", "mid", "high"], value="low", label="level")

        with gr.Row():
            # 执行
            btn = gr.Button("run")
            btn.click(fn=process, inputs=[input_image, level_drop], outputs=ouput_image)

    demo.launch()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
