import pynlpir

pynlpir.open()

str = '张三是我的老师'

print(pynlpir.segment(str))

print(pynlpir.segment(str, pos_english=False))  # 把词性标注语言变更为汉语

print(pynlpir.segment(str, pos_tagging=False))  # 使用pos_tagging来关闭词性标注