## 基于 LSTM 的多分类文本情感分析
### 文本情感分析系统设计
无论是在网上获取的文本(在微博、淘宝上的评论等)还是微信、QQ 上的日常对话，都是一些原生数据，在计算机看来是十分杂乱的。所以我们需要先对文本数据进行分词和词语整理，以及词频统计，同时过滤一些不常见的符号和生僻字，即首先需要一个文本预处理模块。其次，对于文本情感的分析我打算将其应用到日常的对话中，所以需要一个
模拟的对话情景来检验模型的泛化能力。所以还要设计一个模拟对话模块。最终该文本情感分析系统的概述流程如下图所示：
- ![image](https://github.com/Laplacemon/NLP-LSTM/blob/main/img/system_describe.png)

### LSTM神经网络
长短期记忆 (long short-term memory, LSTM)神经网络引入了 3 个门，输入门(input gate)、遗忘门 (forget gate)和输出门 (output gate)，以及与隐藏状态形状相同的记忆细胞，从而记录额外的信息。下图是标准的 LSTM 网络：
- ![image](https://github.com/Laplacemon/NLP-LSTM/blob/main/img/LSTM-Nework.png)
