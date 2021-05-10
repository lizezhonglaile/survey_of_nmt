# 神经机器翻译技术综述
* [(一) 前言](#history)
    * [问题起源](#problem_definition)
    * [解决意义](#problem_value)
* [(二) 主要技术范式](#methods_machine_translation)
    * [基于规则的方法](#rmt)
    * [基于实例的方法](#ebmt)
    * [基于统计的方法](#statistical_machine_translation)
    * [基于深度学习的方法](#neural_machine_translation)
* [(三) 神经机器翻译基础：概念和模型](https://github.com/lizezhonglaile/survey_of_nmt/blob/main/chapter-3-basic.md)
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
* [(四) 神经机器翻译前沿：挑战和现状](https://github.com/lizezhonglaile/survey_of_nmt/blob/main/chapter-4-challenge.md)
    * [受限词表](#oov)
        * [子词模型](#subword_nmt)
        * [字符模型](#char_nmt)
        * [混合模型](#hybrid_nmt)
        * [字节模型](#byte_nmt)
    * [忠实度](#loyal_translation)
    * [资源稀缺](#low_resource)
        * [数据增强](#data_argument)
        * [单语语料利用](#monolingual_exploit)
        * [预训练](#pretraining)
        * [领域迁移](#domain_transfer)
        * [多语言迁移](#multi_lingual_transfer)
    * [知识融合](#knowledge_merge)
        * [词汇知识](#word_knowledge_merge)
        * [短语知识](#phrase_knowledge_merge)
        * [句法知识](#syntax_merge)
        * [世界知识](#world_knowledge_merge)
    * [解码问题](#decoding_space)
        * [双向解码](#decoding_bi)
        * [层次解码](#decoding_heir)     
        * [非自回归解码](#decoding_parallel)
        * [二次解码](#decoding_mutli_pass)
    * [模型一致性](#model_consistence)
    * [模型可控性](#model_control)
    * [模型可解释性](#understanding)
        * [显式建模](#explict_model)
        * [可视化](#visualize)
    * [模型健壮性](#model_robust)
    * [情感和文学性](#elegent_nmt)
* [(五) 神经机器翻译的未来](#future)
* [(六) 效果评测](#evaluation)
    * [公开数据集](#corpus)
    * [人工评测](#human_eval)
    * [自动评测](#auto_eval)
* [(七) 开源工具](#open_tool)
    * [翻译模型](#translation_tool)
    * [数据预处理](#preprocess_tool)
    * [分析和评测](#eval_analysis_tool)
* [(八) 参考文献](#refence)

<h2 id="history">(一) 前言</h2>

<h3 id="problem_definition">问题起源</h3>
　　“机器翻译”是指利用计算机实现自然语言的自动翻译的技术。更广义的翻译包含了多模态机器翻译，即输入出了包括文本信息，还可能包括语音，图像，视频等多模态信息。大体上，机器翻译的发展可以分为一下几个阶段：早期探索时期(1933-1956)；第一次热潮时期(1956-1966)；商用的基于规则时期(1967-2007)；统计机器学习时期(1993-2016)；神经网络机器翻译时期，2013至今。
  
　　早在1949 年,Weaver 发表的以《翻译》为题的备忘录中就提出:“当我阅读一篇用俄语写的文章的时候,我可以说,这篇文章实际上是用英语写的,只不过它是用另外一种奇怪的符号编了码而已,当我在阅读时,我是在进行解码。”这实际上就是基于信源信道思想的统计机器翻译方法的萌芽。
  
　　在1954年，Georgetown University和IBM 一起展示了能够翻译49个俄语句子的系统。其实该系统只能翻译250个词，总共就只有6个转写规则。然而就是这样一个系统，引发了1956年-1966年期间的巨大的机器翻译泡沫，学者们纷纷开始预测机器翻译很快会得到彻底的解决。经过10年的发展，在1964年政府成立了一个自动语言处理顾问委员会Automatic Language Processing Advisory Committee (ALPAC)。经过两年的调查，ALPAC发表了其著名的机器翻译研究现状报告，其结论是机器翻译进展缓慢，质量糟糕，价格昂贵，且看不到未来。这一报告直接使得机器翻译陷入了长达十年的寒冬。在这个寒冬中，失去政府资助的机器翻译研究人员只好转向定制商用系统，基于转写文法（或称为基于规则的方法，Rule-based）的方法往往对特定的定制领域有较好的效果，因此统计方法几乎被抛弃。
    
　　随着数字化文本语料越来越多，基于语料的机器翻译逐渐占据主流。1993年，IBM的 Brown et al. 发表了The mathematics of statistical machine translation: Parameter estimation。这篇文章奠定了此后20年机器翻译的基础。然而，在词到词模型出现的前10年，并没有获得很大的成功。其原因主要是翻译单元粒度太小，利用上下文的能力过弱。统计机器翻译的真正崛起，始于Franz Och在2003年的两篇文章Statistical phrase-based translation和Minimum error rate training in statistical machine translation。这篇文章提出了基于短语的翻译模型和最小错误率训练方法。此后直到2015，2016年，这两种方法都是机器翻译的主流方法。2004年，Franz Och加入谷歌，并领导了谷歌翻译的开发。2006年，谷歌翻译作为一个免费服务正式发布，并带来了统计机器翻译研究的一大波热潮。截止2015年，谷歌翻译已经支持了超过100种语言。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/mt_history.jpg" width="75%" height="75%"></div>
<div align="center">机器翻译发展史</div>

<h3 id="problem_value">解决意义</h3>

<h2 id="methods_machine_translation">(二) 主要技术范式</h2>
从学术上讲，似乎一切混合模型都是在灌水；但从工业界上讲，所有单模型都是纸上谈兵。
<h3 id="rmt">基于规则的方法</h3>

<h3 id="ebmt">基于实例的方法</h3>

Makoto Nagao. A framework of a mechanical translation between japanese and english by analogy principle. Artificial and human intelligence, pages 351–354, 1984.

<h3 id="statistical_machine_translation">基于统计的方法</h3>

1993年，IBM的 Brown et al. 发表了The mathematics of statistical machine translation: Parameter estimation。这篇文章奠定了此后20年机器翻译的基础。这篇文章将机器翻译描述为一个信道模型

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/phrase_smt.png" width="55%" height="55%"></div>
        
<h4 id="word_smt">词翻译模型</h4>

<h4 id="phrase_smt">短语翻译模型</h4>

<h3 id="smt_tutorials">基于深度学习的方法</h3>
与统计机器翻译的离散表示方法不同，神经机器翻译采用连续空间表示方法（Continuous  Space Representation）表示词语、短语和句子。在翻译建模上，不需要词对齐、翻译规则抽取等统计机器翻译的必要步骤，完全采用神经网络完成从源语言到目标语言的映射。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/deeplearning.png" width="75%" height="75%"></div>
<div align="center">深度学习提供新思路</div>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/smt_vs_nmt.png" width="50%" height="50%"></div>


<h2 id="future">(五) 神经机器翻译的未来</h2>

<h2 id="evaluation">(六) 效果评测</h2>

<h3 id="corpus">数据集</h3>

<h3 id="human_eval">人工评测</h3>

<h3 id="auto_eval">自动评测</h3>

K. Papineni, S. Roukos, T.Ward,W. Zhu, Bleu: A method for automatic evaluation of machine translation, in: Proceedings of ACL, 2002.

<h2 id="open_tool">(七) 开源工具</h2>

<h3 id="translation_tool">翻译模型</h3>

M. Ott, S. Edunov, A. Baevski, A. Fan, S. Gross, N. Ng, D. Grangier, M. Auli, fairseq: A fast, extensible toolkit for sequence modeling, in: Proceedings of NAACL-HLT (Demonstrations), 2019, pp. 48–53.

Guillaume Klein, etc, OpenNMT: Open-Source Toolkit for Neural Machine Translation, in: Proc. ACL, 2017

A. Vaswani, etc., Tensor2Tensor for neural machine translation, in: Proceedings of AMTA, 2018, pp. 193–199.

Z. Tan, J. Zhang, X. Huang, G. Chen, S.Wang, M. Sun, H. Luan, Y. Liu, THUMT: An open-source toolkit for neural machine translation, in: Proceedings of AMTA, 2020, pp. 116–122.

<h3 id="preprocess_tool">数据预处理</h3>

<h3 id="eval_analysis_tool">分析和评测</h3>

Ondřej Klejch, etc. MT-ComparEval: Graphical evaluation interface for Machine Translation development. The Prague Bulletin of Mathematical Linguistics, NUMBER 104 OCTOBER 2015 63–74

G. Neubig, Z.-Y. Dou, J. Hu, P. Michel, D. Pruthi, X. Wang, comparemt: A tool for holistic comparison of language generation systems, in: Proceedings of NAACL-HLT (Demonstrations), 2019, pp. 35–41.

<h2 id="refence">(八) 参考文献</h2>

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

