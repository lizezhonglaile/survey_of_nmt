* [(三) 神经机器翻译基础：概念和模型](#neural_machine_translation)
    * [词嵌入](#word_representation)
        * [单语言词嵌入](#monolingual_word_representation)
        * [跨语言词嵌入](#cross_word_representation)
    * [翻译建模：encoder-decoder框架](#encoder-decoder)
        * [基于RNN的序列到序列模型](#rnn_seq2seq)  
        * [注意力机制](#attention)
        * [transformer模型](#transformer)
        * [其他encoder-decoder模型](#other_seq2seq)
    * [翻译模型训练](#learning_nmt)
    * [翻译解码](#decoding_nmt)
    * [多语言](#multilingual_nmt)
    * [多模态](#multimodal_nmt)


<h2 id="neural_machine_translation">(三) 神经机器翻译基础：概念和模型</h2>
本章围绕机器翻译三个最核心问题展开，即翻译建模，模型训练和解码。在此之前，有必要先介绍一下词表示学习，它是整个基于深度学习的自然语言处理基础，也是神经网络机器翻的基础。然后围绕核心问题，介绍翻译模型的几个里程碑意义的进展，模型训练的目标函数选择以及基于 Beam search 的翻译解码算法。最后，对于近几年比较新的多语言，多模态也进行一个简单的介绍。
<h3 id="word_representation">词嵌入</h3>

<h4 id="monolingual_word_representation">单语言词嵌入</h4>

<h4 id="cross_word_representation">跨语言词嵌入</h4>

M. Artetxe, G. Labaka, E. Agirre, Learning bilingual word embeddings with (almost) no bilingual data, in: Proceedings of ACL, 2017, pp. 451–462.
    
<h3 id="encoder-decoder">翻译建模：encoder-decoder框架</h3>
<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/seq2seq.png" width="55%" height="55%"></div>
<div align="center">seq2seq模型</div>

<h4 id="rnn_seq2seq">基于RNN的序列到序列模型</h4>

<h4 id="attention">注意力机制</h3>
<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/attention.png" width="55%" height="55%"></div>
<div align="center">注意力模型</div>

<h4 id="transformer">transformer模型</h3>
变形金刚

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/transformer.jpg" width="75%" height="75%"></div>
<div align="center">transformer模型</div>

<h4 id="other_seq2seq">其他encoder-decoder模型</h4>
CNN
GCN
Graph Transformer

<h3 id="learning_nmt">翻译模型学习</h3>
损失函数
其他学习范式，无监督，半监督

<h3 id="decoding_nmt">翻译解码</h3>
beam search

<h3 id="multilingual_nmt">多语言</h3>
多语言机器翻译，区别于通常一种语言到另外一种语言的一对一翻译，能够采用一个模型完成多种语言之间翻译。基于神经网络的多语言机器翻译源于序列到序列学习和多任务学习，从类型上可以分为单语到多语翻译、多语到单语翻译，以及多语到多语翻译。
        
<h4 id="many_to_one">多对一</h4>
多语到单语翻译是源语言有多个，而目标语言只有一个的机器翻译方法。典型工作为 Zoph 和Knight[58]提出的多语到单语翻译方法。该方法有两个源语言，分别对应一个编码器，在注意力机制上采用了多源语言注意力机制，这是对 Luong 等人]出的局部注意力的改进。在 t 时刻分别从两个源语言得到上下文向量 和 ，同时应用在解码中。实验采用 WMT 2014 语料，当源语言为法语、德语，目标语言为英语时，相比一对一的翻译，提高了 4.8 个 BLEU 值；当源语言为英语、法语，目标语言为德语时，则提高 1.1 个 BLEU 值。可以看出，源语言之间的差异对该方法影响很大。除此之外，对每个源语言采用单独的注意力机制，计算复杂度较高。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/many-to-one.png" width="50%" height="50%"></div>
<div align="center">(a) 多对一</div>

<h4 id="one_to_many">一对多</h4>
单语到多语翻译是源语言只有一个，而目标语言有多个的机器翻译方法。Dong 等人首次将多任务学习引入序列到序列学习，实现了一种单语到多语的神经机器翻译方法。该方法在编码器解码器上增加了多任务学习模型，源语言采用一个编码器，每个目标语言单独采用一个解码器，每个解码器都有自己的注意力机制，但是共享同一个编码器。实验采用欧洲语料库，源语言为英语，目标语言分别为法语、西班牙语、荷兰语、葡萄牙语。实验结果显示单语到多语的机器翻译效果均高于英语到其他语言之间的单独翻译，在多数语言对上均提高 1 个 BLEU 值以上。这种方法共享源语言编码器，能够提高资源稀缺语言对翻译质量。不足之处是每个解码器都拥有单独的注意力机制，计算复杂度较高，限制了在大规模语言对上的应用。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/multilingual-baidu.png" width="55%" height="55%"></div>
<div align="center">(b) 一对多</div>

<h4 id="many_to_many">多对多</h4>
多语到多语翻译是源语言和目标语言均有多个的机器翻译方法，可以实现多种语言之间互译。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/many-to-many.png" width="55%" height="55%"></div>
<div align="center">(c) 多对多</div>

<h3 id="multimodal_nmt">多模态</h3>
多模态神经机器翻译利用的资源不限于文本，目前研究主要集中在利用图像信息提高神经机器翻译效果。这类方法通常采用两个编码器，一个编码器对文本信息编码，与普通的神经机器翻译相同；另外一个编码器对图像信息编码。在解码时，通过注意力机制将不同模态的信息应用在翻译中。
