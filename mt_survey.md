# 机器翻译技术调研
* [背景介绍](#history)
* [主要技术范式](#methods_machine_translation)
    * [直接机器翻译](#direct_mt)
    * [句法转换机器翻译](#syntax_tranduct_mt)
    * [基于中间语言的机器翻译](#interlingua_mt)
    * [基于规则的机器翻译](#rmt)
    * [基于实例的机器翻译](#ebmt)
    * [基于统计的机器翻译](#statistical_machine_translation)
    * [基于神经网络的机器翻译](#neural_machine_translation)
* [神经机器翻译基础](#neural_machine_translation)
    * [seq2seq模型](#seq2seq)
    * [注意力模型](#atention)
    * [transformer模型](#transformer)
    * [短语神经模型](#phrase_nmt)
    * [融合句法的神经模型](#syntax_num)
    * [多语言](#multilingual_nmt)
    * [多模态](#multimodal_nmt)
    * [预训练模型](#pretraining)
    * [非平行语料利用](#beyond_parallel)
    * [解码](#decoding_nmt)
* [神经机器翻译前沿: 挑战和进展](#challenges)
    * [稀疏词](#oov)
    * [忠实度](#less_over_translation)
    * [低资源](#low_resource)
    * [知识融合](#knowleding_merge)
    * [跨领域跨语言和跨模态的迁移学习](#transfer)
    * [模型可解释性](#understanding)
    * [诗歌翻译](#poem_nmt)
* [效果评测](#evaluation)
    * [人工评测](#human_eval)
    * [自动评测](#auto_eval)
    * [公开数据集和评测](#public_eval)
* [参考文献](#refence)

<h2 id="history">背景介绍</h2>
　　“机器翻译”是指利用计算机实现自然语言的自动翻译的技术。更广义的翻译包含了多模态机器翻译，即输入出了包括文本信息，还可能包括语音，图像，视频等多模态信息。大体上，机器翻译的发展可以分为一下几个阶段：早期探索时期(1933-1956)；第一次热潮时期(1956-1966)；商用的基于规则时期(1967-2007)；统计机器学习时期(1993-2016)；神经网络机器翻译时期，2013至今。
  
　　早在1949 年,Weaver 发表的以《翻译》为题的备忘录中就提出:“当我阅读一篇用俄语写的文章的时候,我可以说,这篇文章实际上是用英语写的,只不过它是用另外一种奇怪的符号编了码而已,当我在阅读时,我是在进行解码。”这实际上就是基于信源信道思想的统计机器翻译方法的萌芽。
  
　　在1954年，Georgetown University和IBM 一起展示了能够翻译49个俄语句子的系统。其实该系统只能翻译250个词，总共就只有6个转写规则。然而就是这样一个系统，引发了1956年-1966年期间的巨大的机器翻译泡沫，学者们纷纷开始预测机器翻译很快会得到彻底的解决。经过10年的发展，在1964年政府成立了一个自动语言处理顾问委员会Automatic Language Processing Advisory Committee (ALPAC)。经过两年的调查，ALPAC发表了其著名的机器翻译研究现状报告，其结论是机器翻译进展缓慢，质量糟糕，价格昂贵，且看不到未来。这一报告直接使得机器翻译陷入了长达十年的寒冬。在这个寒冬中，失去政府资助的机器翻译研究人员只好转向定制商用系统，基于转写文法（或称为基于规则的方法，Rule-based）的方法往往对特定的定制领域有较好的效果，因此统计方法几乎被抛弃。
    
　　随着数字化文本语料越来越多，基于语料的机器翻译逐渐占据主流。1993年，IBM的 Brown et al. 发表了The mathematics of statistical machine translation: Parameter estimation。这篇文章奠定了此后20年机器翻译的基础。然而，在词到词模型出现的前10年，并没有获得很大的成功。其原因主要是翻译单元粒度太小，利用上下文的能力过弱。统计机器翻译的真正崛起，始于Franz Och在2003年的两篇文章Statistical phrase-based translation和Minimum error rate training in statistical machine translation。这篇文章提出了基于短语的翻译模型和最小错误率训练方法。此后直到2015，2016年，这两种方法都是机器翻译的主流方法。2004年，Franz Och加入谷歌，并领导了谷歌翻译的开发。2006年，谷歌翻译作为一个免费服务正式发布，并带来了统计机器翻译研究的一大波热潮。截止2015年，谷歌翻译已经支持了超过100种语言。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/mt_history.jpg" width="75%" height="75%"></div>
<div align="center">机器翻译发展史</div>

<h2 id="methods_machine_translation">主要技术范式</h2>

<h3 id="direct_mt">直接机器翻译</h3>

<h3 id="syntax_tranduct_mt">句法转换机器翻译</h3>

<h3 id="interlingua_mt">基于中间语言的机器翻译</h3>

<h3 id="rmt">基于规则的机器翻译</h3>

<h3 id="ebmt">基于实例的机器翻译</h3>

<h3 id="statistical_machine_translation">基于统计的机器翻译</h3>

1993年，IBM的 Brown et al. 发表了The mathematics of statistical machine translation: Parameter estimation。这篇文章奠定了此后20年机器翻译的基础。这篇文章将机器翻译描述为一个信道模型（事实

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/phrase_smt.png" width="55%" height="55%"></div>

<h3 id="smt_tutorials">基于神经网络的机器翻译</h3>
<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/deeplearning.png" width="75%" height="75%"></div>
<div align="center">深度学习提供新思路</div>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/smt_vs_nmt.png" width="50%" height="50%"></div>

<h2 id="neural_machine_translation">神经机器翻译基础</h2>

<h3 id="seq2seq">seq2seq模型</h3>
<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/seq2seq.png" width="55%" height="55%"></div>
<div align="center">seq2seq模型</div>

<h3 id="atention">注意力模型</h3>
<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/attention.png" width="55%" height="55%"></div>
<div align="center">注意力模型</div>

<h3 id="transformer">transformer模型</h3>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/transformer.jpg" width="75%" height="75%"></div>
<div align="center">transformer模型</div>

<h3 id="phrase_nmt">短语神经模型</h3>

<h3 id="syntax_num">融合句法的神经模型</h3>

<h3 id="multilingual_nmt">多语言</h3>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/many-to-one.png" width="50%" height="50%"></div>
<div align="center">(a) 多对一</div>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/multilingual-baidu.png" width="55%" height="55%"></div>
<div align="center">(b) 一对多</div>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/many-to-many.png" width="55%" height="55%"></div>
<div align="center">(c) 多对多</div>

<h3 id="multimodal_nmt">多模态</h3>

<h3 id="pretraining">非平行语料利用</h3>

<h3 id="beyond_parallel">预训练模型</h3>

<h3 id="decoding_nmt">解码</h3>

<h4>beam search 算法</h4>

<h4>双向解码</h4>

<h4>层次解码</h4>

<h4>曝光偏置问题</h4>

<h2 id="challenges">神经机器翻译前沿: 挑战和进展</h2>

<h3 id="oov">稀疏词问题</h3>

<h3 id="less_over_translation">忠实度</h3>
忠实度，即“信达雅”中的信，是翻译最起码的要求。不忠实主要表现为漏翻译和过翻译

<h3 id="low_resource">低资源</h3>

<h3 id="knowleding_merge">知识融合</h3>

<h3 id="transfer">跨领域跨语言和跨模态的迁移学习</h3>

<h3 id="understanding">模型可解释性</h3>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/black-box.jpg" width="55%" height="55%"></div>
<div align="center">transformer模型</div>

<h3 id="poem_nmt">诗歌翻译</h3>

<h2 id="evaluation">效果评测</h2>

<h3 id="human_eval">人工评测</h3>

<h3 id="auto_eval">自动评测</h3>

<h3 id="public_eval">公开数据集和评测</h3>

<h2 id="refence">参考文献</h2>

* Peter E. Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and Robert L. Mercer. 1993. [The Mathematics of Statistical Machine Translation: Parameter Estimation](http://aclweb.org/anthology/J93-2003). *Computational Linguistics*. ([Citation](https://scholar.google.com/scholar?cites=2259057253133260714&as_sdt=2005&sciodt=0,5&hl=en): 5,218)
* Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P02-1040). In *Proceedings of ACL 2002*. ([Citation](https://scholar.google.com/scholar?cites=9019091454858686906&as_sdt=2005&sciodt=0,5&hl=en): 10,700)
* Philipp Koehn, Franz J. Och, and Daniel Marcu. 2003. [Statistical Phrase-Based Translation](http://aclweb.org/anthology/N03-1017). In *Proceedings of NAACL 2003*. ([Citation](https://scholar.google.com/scholar?cites=11796378766060939113&as_sdt=2005&sciodt=0,5&hl=en): 3,713)
* Franz Josef Och. 2003. [Minimum Error Rate Training in Statistical Machine Translation](http://aclweb.org/anthology/P03-1021). In *Proceedings of ACL 2003*. ([Citation](https://scholar.google.com/scholar?cites=15358949031331886708&as_sdt=2005&sciodt=0,5&hl=en): 3,115)
* David Chiang. 2007. [Hierarchical Phrase-Based Translation](http://aclweb.org/anthology/J07-2003). *Computational Linguistics*. ([Citation](https://scholar.google.com.hk/scholar?cites=17074501474509484516&as_sdt=2005&sciodt=0,5&hl=en): 1,235)
* Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. [Sequence to Sequence Learning
with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf). In *Proceedings of NIPS 2014*. ([Citation](https://scholar.google.com/scholar?cites=13133880703797056141&as_sdt=2005&sciodt=0,5&hl=en): 9,432)
* Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf). In *Proceedings of ICLR 2015*. ([Citation](https://scholar.google.com/scholar?cites=9430221802571417838&as_sdt=2005&sciodt=0,5&hl=en): 10,479)
* Diederik P. Kingma, Jimmy Ba. 2015. [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980). In *Proceedings of ICLR 2015*. ([Citation](https://scholar.google.com/scholar?cites=16194105527543080940&as_sdt=2005&sciodt=0,5&hl=en): 37,480)
* Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf). In *Proceedings of ACL 2016*. ([Citation](https://scholar.google.com/scholar?cites=1307964014330144942&as_sdt=2005&sciodt=0,5&hl=en): 1,679)
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). In *Proceedings of NIPS 2017*. ([Citation](https://scholar.google.com/scholar?cites=2960712678066186980&as_sdt=2005&sciodt=0,5&hl=en): 6,112)

