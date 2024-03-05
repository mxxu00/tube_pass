import os
import paddlehub as hub
import cv2
from operator import itemgetter

# ------ CUDA 设置 ------ #
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 获取GPU设备，不使用GPU时注释
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 设置GPU编号，不使用GPU时注释

# ------ 过滤ocr结果 ------ #
def filter_ocr_results(test_img_path, ocr_results):
    for img_path, result in zip(test_img_path, ocr_results):  # zip 合并两个list
        data = result['data']
        data = sorted(data, key=itemgetter('confidence'), reverse=True) # 按照置信度对list中的dict进行排序，置信度高的排在前面
        for information in data: # 遍历全部识别结果、置信度（即识别结果正确的可能性）、识别框的位置（四个角点的坐标）
            print(information)

if __name__ == "__main__":
    # ------ 更改测试数据 ------ #
    data_path = 'D:/chengguo/1.jpg' # 获取图像文件的相对路径，可以是单张图片，也可以是图片数据集

    # 接将path下所有图片加载
    img_path_list = []
    basedir = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件的绝对路径
    input_path_dir = data_path  # 文件夹路径
    img_path_list.append(input_path_dir)
    # 读取测试文件夹中的图片路径
    np_images = [cv2.imread(image_path) for image_path in img_path_list]

    # OCR图片的保存路径，默认设为
    output_path = basedir

    # 输出图像文件路径和结果输出路径
    print("input_path_dir: " + input_path_dir)
    print("output_path: " + output_path)

    # 加载预训练模型
    ocr = hub.Module(name="chinese_ocr_db_crnn_server") # 在不使用GPU的情况下可以设置 enable_mkldnn=True 加速模型

    # 进行OCR识别
    ocr_results = ocr.recognize_text(
        images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
        use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
        output_dir=output_path,  # 识别结果图片文件的保存路径；
        visualization=True,  # 是否将识别结果保存为图片文件；
        box_thresh=0.0001,  # 检测文本框置信度的阈值，越小识别的内容越多，但也更可能识别出不相关的内容，不可设置为0；
        text_thresh=0.5)  # 识别中文文本置信度的阈值，是指将内容识别为中文字符的阈值参数，由于也需要对英文和数字进行识别，该值不宜过小；
    # 过滤识别结果
    filter_ocr_results(img_path_list, ocr_results)