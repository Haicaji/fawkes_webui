import gradio as gr
import numpy as np
from PIL import Image
import time
import os

import fawkes.protection as fp

def process(input_image, level):
    # 获取当前所处文件夹
    cur_path = os.path.abspath(__file__)
    cur_path = os.path.dirname(cur_path)
    # 获取文件名
    input_image_name = os.path.basename(input_image)
    # 获取后缀
    input_image_suffix = os.path.splitext(input_image_name)[1]

    # 复制图像到imgs文件夹
    copy_command = f'copy "{input_image}" "{cur_path}\\imgs"'
    os.system(copy_command)

    # 获取当前时间
    now_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    tmp_image_name = f"{cur_path}\\imgs\\{input_image_name.replace(input_image_suffix, f'{now_time}_input{input_image_suffix}')}"
    output_img_name = f"{cur_path}\\imgs\\{input_image_name.replace(input_image_suffix, f'{now_time}_output{input_image_suffix}')}"

    # 重命名文件
    os.rename(f"{cur_path}\\imgs\\{input_image_name}", tmp_image_name)

    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception as e:
        pass

    try:
        protector = fp.Fawkes(
            feature_extractor="arcface_extractor_0",
            gpu="0",
            batch_size=1,
            mode=level
        )

        protector.run_protection(
            [tmp_image_name],
            th=0.01,
            sd=1e6,
            lr=2,
            max_step=1000,
            batch_size=1,
            format=input_image_suffix[1:] if input_image_suffix[1:]!='jpg' else 'jpeg',
            separate_target=False,
            debug=False,
            no_align=False
        )


    except Exception as e:
        print(e)
        print("Error")
        # 删除临时文件
        os.remove(tmp_image_name)
        return input_image

    # 重命名文件
    os.rename(tmp_image_name.replace("input", f"input_cloaked"), 
              output_img_name)
    os.remove(tmp_image_name)

    return output_img_name

def main():
    with gr.Blocks() as demo:
        # 图片
        with gr.Row():
            # 上传图片
            input_image = gr.Image(height=500, source='upload', type="filepath")
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
