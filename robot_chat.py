import requests
from predict import predict

q_yun_url = "http://api.qingyunke.com/api.php?key=free&appid=0&msg="
s_zhi_url = "https://api.ownthink.com/bot?userid=user&spoken="
s_zhi_url_g = "https://api.ownthink.com/bot?spoken="

text = "今天天气真好！我好开心哦"


def chat_print(speaker, content, pred_str=None, print_type='left'):
    """
    在控制台按对话左右对齐格式打印文本
    :param speaker:
    :param content:
    :param print_type:
    :return:
    """
    len_speaker = len(speaker) + 1
    len_c = len(content)
    box_width = 25
    rows = len_c // box_width + 2 + int(len_c % box_width > 0)
    cols = len_speaker + box_width + 1
    str_list = (['一'] * (cols - 1) + ['\n']) * rows
    # 使用中文全角空格，与一个文字一般大
    if print_type == "left":
        str_list[cols:cols * 2] = list(speaker + '：' + content[:box_width] + '\n')
        for i in range(3, rows):
            c_list = ['　'] * len_speaker + list(content[box_width * (i - 2):box_width * (i - 1)])
            if len(c_list) < cols - 1:
                c_list += ["　"] * (cols - 1 - len(c_list))
            str_list[cols * (i - 1):cols * i] = c_list + ['\n']
        str_list.insert(-(box_width+7), pred_str + "\n")
        print("".join(str_list))
    else:
        str_list[cols:cols * 2] = list(content[:box_width] + '：') + list(speaker + '\n')
        for i in range(3, rows):
            c_list = list(content[box_width * (i - 2):box_width * (i - 1)]) + ['　'] * len_speaker
            if len(c_list) < cols - 1:
                c_list += ["　"] * (cols - 1 - len(c_list))

            str_list[cols * (i - 1):cols * i] = c_list + ['\n']
        str_list.insert(-(box_width+7), pred_str + "\n")
        str_prints = "".join(str_list)
        for str_list_item in str_prints.split('\n'):
            # 指定填充为全角空格
            print("{:　>{size}}".format(str_list_item, size=40+box_width))


n = 0
while n < 50:
    q_yun_respond = requests.get(q_yun_url + text).json().get('content')
    # q_yun_respond = requests.post(s_zhi_url + text).json()['data']['info']['text']
    chat_print("青云机器人", q_yun_respond, pred_str=predict(q_yun_respond)[:31])
    text = q_yun_respond
    n += 1
    s_zhi_respond = requests.post(s_zhi_url + text).json()['data']['info']['text']
    # s_zhi_respond = requests.get(s_zhi_url_g + text).json()['data']['info']['text']
    chat_print("思知机器人", s_zhi_respond,pred_str=predict(s_zhi_respond)[:31], print_type="right")
    text = s_zhi_respond
