# 机器翻译技术调研
* [前言](#history)
* [主要技术范式](#methods_machine_translation)
    * [直接机器翻译](#direct_mt)
    * [句法转换机器翻译](#syntax_tranduct_mt)
    * [基于中间语言的机器翻译](#interlingua_mt)
    * [基于规则的机器翻译](#rmt)
    * [基于实例的机器翻译](#ebmt)
    * [基于统计的机器翻译](#statistical_machine_translation)
        * [词统计机器翻译](#word_smt)
        * [短语统计机器翻译](#phrase_smt)
        * [层次短语统计机器翻译](#heir_smt)
    * [基于神经网络的机器翻译](#neural_machine_translation)
* [神经机器翻译基本概念](#neural_machine_translation)
    * [词表示学习](#word_representation)
    * [句子表示学习](#sent_representation)
    * [seq2seq模型](#seq2seq)
    * [注意力模型](#atention)
    * [transformer](#transformer)
    * [解码](#decoding_nmt)
        * [单向解码](#decoding_mono)
        * [双向解码](#decoding_bi)
        * [层次解码](#decoding_heir)
        * [并行解码](#decoding_parallel)
        * [曝光偏置问题](#decoding_exposuer)
    * [多语言](#multilingual_nmt)
    * [多模态](#multimodal_nmt)
* [神经机器翻译挑战和进展](#challenges)
    * [稀疏词](#oov)
        * [细粒度神经模型](#char_nmt)
    * [忠实度](#less_over_translation)
    * [资源稀缺](#low_resource)
        * [数据增强](#data_argument)
        * [单语语料利用](#monolingual_exploit)
        * [预训练模型](#pretraining)
        * [领域迁移](#domain_transfer)
        * [多语言迁移](#multi_lingual_transfer)
    * [语言学知识融合](#knowledge_merge)
    * [模型可解释性](#understanding)
    * [情感和文学性](#elegent_nmt)
* [效果评测](#evaluation)
    * [公开数据集和评测](#public_eval)
    * [人工评测](#human_eval)
    * [自动评测](#auto_eval)
* [参考文献](#refence)

<h2 id="history">前言</h2>
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

1993年，IBM的 Brown et al. 发表了The mathematics of statistical machine translation: Parameter estimation。这篇文章奠定了此后20年机器翻译的基础。这篇文章将机器翻译描述为一个信道模型

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/phrase_smt.png" width="55%" height="55%"></div>

<h4 id="word_smt">词统计机器翻译</h4>

<h4 id="phrase_smt">短语统计机器翻译</h4>

<h4 id="heir_smt">层次短语统计机器翻译</h4>

<h3 id="smt_tutorials">基于神经网络的机器翻译</h3>
与统计机器翻译的离散表示方法不同，神经机器翻译采用连续空间表示方法（Continuous  Space Representation）表示词语、短语和句子。在翻译建模上，不需要词对齐、翻译规则抽取等统计机器翻译的必要步骤，完全采用神经网络完成从源语言到目标语言的映射。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/deeplearning.png" width="75%" height="75%"></div>
<div align="center">深度学习提供新思路</div>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/smt_vs_nmt.png" width="50%" height="50%"></div>

<h2 id="neural_machine_translation">神经机器翻译基本概念</h2>

<h3 id="word_representation">词表示学习</h3>

<h3 id="sent_representation">句子表示学习</h3>
    
<h3 id="seq2seq">seq2seq模型</h3>
<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/seq2seq.png" width="55%" height="55%"></div>
<div align="center">seq2seq模型</div>

<h3 id="atention">注意力模型</h3>
<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/attention.png" width="55%" height="55%"></div>
<div align="center">注意力模型</div>

<h3 id="transformer">transformer模型</h3>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/transformer.jpg" width="75%" height="75%"></div>
<div align="center">transformer模型</div>

<h3 id="decoding_nmt">解码</h3>

<h4 id="decoding_mono">单向解码</h4>

<h4 id="decoding_bi">双向解码</h4>

<h4 id="decoding_heir">层次解码</h4>

<h4 id="decoding_parallel">并行解码</h4>

<h4 id="decoding_exposuer">曝光偏置问题</h4>

<h3 id="multilingual_nmt">多语言</h3>
多语言机器翻译，区别于通常一种语言到另外一种语言的一对一翻译，能够采用一个模型完成多种语言之间翻译。基于神经网络的多语言机器翻译源于序列到序列学习和多任务学习，从类型上可以分为单语到多语翻译、多语到单语翻译，以及多语到多语翻译。

<h4>一对多</h4>
单语到多语翻译是源语言只有一个，而目标语言有多个的机器翻译方法。Dong 等人首次将多任务学习引入序列到序列学习，实现了一种单语到多语的神经机器翻译方法。该方法在编码器解码器上增加了多任务学习模型，源语言采用一个编码器，每个目标语言单独采用一个解码器，每个解码器都有自己的注意力机制，但是共享同一个编码器。实验采用欧洲语料库，源语言为英语，目标语言分别为法语、西班牙语、荷兰语、葡萄牙语。实验结果显示单语到多语的机器翻译效果均高于英语到其他语言之间的单独翻译，在多数语言对上均提高 1 个 BLEU 值以上。这种方法共享源语言编码器，能够提高资源稀缺语言对翻译质量。不足之处是每个解码器都拥有单独的注意力机制，计算复杂度较高，限制了在大规模语言对上的应用。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/multilingual-baidu.png" width="55%" height="55%"></div>
<div align="center">(b) 一对多</div>

<h4>多对一</h4>
多语到单语翻译是源语言有多个，而目标语言只有一个的机器翻译方法。典型工作为 Zoph 和Knight[58]提出的多语到单语翻译方法。该方法有两个源语言，分别对应一个编码器，在注意力机制上采用了多源语言注意力机制，这是对 Luong 等人]出的局部注意力的改进。在 t 时刻分别从两个源语言得到上下文向量 和 ，同时应用在解码中。实验采用 WMT 2014 语料，当源语言为法语、德语，目标语言为英语时，相比一对一的翻译，提高了 4.8 个 BLEU 值；当源语言为英语、法语，目标语言为德语时，则提高 1.1 个 BLEU 值。可以看出，源语言之间的差异对该方法影响很大。除此之外，对每个源语言采用单独的注意力机制，计算复杂度较高。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/many-to-one.png" width="50%" height="50%"></div>
<div align="center">(a) 多对一</div>

<h4>多对多</h4>
多语到多语翻译是源语言和目标语言均有多个的机器翻译方法，可以实现多种语言之间互译。

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/many-to-many.png" width="55%" height="55%"></div>
<div align="center">(c) 多对多</div>

<h3 id="multimodal_nmt">多模态</h3>
多模态神经机器翻译利用的资源不限于文本，目前研究主要集中在利用图像信息提高神经机器翻译效果。这类方法通常采用两个编码器，一个编码器对文本信息编码，与普通的神经机器翻译相同；另外一个编码器对图像信息编码。在解码时，通过注意力机制将不同模态的信息应用在翻译中。

<h3 id="pretraining">单语语料利用</h3>
单语语料是一种非常重要的资源，具有数量大、获取方便的优势。在统计机器翻译中，大规模目标语言单语语料可以提供优质的语言模型，对提高翻译流利度起着很重要作用。在神经机器翻译中可以利用的单语语料主要分为目标语言单语语料和源语言单语语料。

目标语言单语语料应用之一是语言模型， Gulcehre 等人提出一种利用大规模单语语料提高神经机器翻译效果的方法。采用单语语料训练神经网络语言模型，将之集成到神经机器翻译中，集成方法分为浅层集成和深层集成。浅层集成方法在解码时，把语言模型作为一种特征用来生成候选词；深层集成方法将神经机器翻译模型、语言模型的隐藏状态连接在一起，通过控制机制动态平衡两种模型对解码的影响，在解码时可以捕捉到语言模型信息。这两种集成方法均可以提高翻译效果，其中深层集成方法效果更为明显 。 此外， Domhan 等人则提出采用多任务学习方法，将神经机器翻译模型和目标语言的语言模型联合训练，以此利用大规模目标语言单语语料。

目标语言单语语料的另一使用方法是 Sennrich 等人提出的训练数据构造方法：回翻译（Back-translation）方法。利用目标语言单语语料构造伪双语数据，并加入到训练语料。这种融合方法对神经机器翻译模型不作改变，方法简单有效，虽然在一定程度上提高了翻译效果，但是效果提升取决于构造数据的质量。

以上研究都是利用目标语言单语语料，Zhang 等人提出了将源语言单语语料应用到神经机器翻译的方法。实现方式有两种，第一种方法同样采用了构造数据思想，在构造方式上通过自学习的方法扩大双语训练语料规模；另外一种方法通过多任务学习增强编码器对源语言的表示质量。这两种方法均能够大幅提升翻译效果。不足之处是源语言单语语料的数量和题材会对翻译模型性能产生影响。

同时利用源语言和目标语言单语语料，主要有Cheng 等人提出的半监督学习方法。基本思想是将自编码引入源语言到目标语言翻译模型和目标语言到源语言翻译模型，通过半监督方法训练双向神经机器翻译，以此利用源语言和目标语言单语语料提高翻译效果。这种方法显著优势是可以同时利用源语言和目标语言的单语语料，不足之处是对单语语料中的未登录词没有处理能力。
此外，Ramachandran 等人提出一种更为简单的方法，将序列到序列模型看作为两个语言模型，通过大规模单语语料分别训练源语言和目标语言的语言模型；神经机器翻译模型的编码器和解码器参数分别由两个语言模型参数初始化；然后利用双语平行语料训练，训练过程中语言模型参数同时调整。


<h3 id="beyond_parallel">预训练模型</h3>

<h2 id="challenges">神经机器翻译挑战和进展</h2>

<h3 id="oov">稀疏词问题</h3>

<h4 id="char_nmt">细粒度神经模型</h4>
字符级神经机器翻译（Character Level NMT）是为了解决未登录词、词语切分、词语形态变化等问题提出的一种神经机器翻译模型，主要特点是减小了输入和输出粒度。
词语编码方案

多数神经机器翻译模型都以词语作为翻译基本单位，存在未登录词、数据稀疏，以及汉语、日语等语言中的分词问题。此外，在形态变化较多的语言中，如英语、法语等语言，以词为处理基本单位时，丢失了词语之间的形态变化、语义信息。如英语单词，“run”，“runs”，“ran”，“running”被认为是四个不同的词，忽略了他们有着共同的前缀“run”。为了解决上述问题，学者们提出了不同的词语编码方案，根据粒度划分可以归为以下两种：

（1）字符编码方案。对于英语、法语等拼音文字来说字符是组成词语的基本单位，在语言处理中能够以字符为单位建模。这方面工作很早就开始研究，比如字符级神经网络语言模型[48]。该方案同时也存在不足，比如编码粒度过小，适合英语、法语等字符数量相近的语言之间的翻译，如果用在英语到汉语翻译上会出现诸多问题。

（2）亚词编码方案。亚词编码方案选用的翻译基本单位介于字符和词语，可以得到两种方案的共同优势。词素的粒度同样介于字符和词语之间，不足之处是跟特定语言相关，限制了应用的通用性。因此，亚词通常采用 BPE 编码（Byte Pair Encoding, BPE）得到[49]，该方案将经常结合的字符组合看作是一个单位，比如词语 “ dreamworks interactive”，可以切分成“dre + am + wo + rks/ in + te + ra + cti + ve”序列，方法简单有效，适应性强。

基于字符的翻译模型


<h3 id="less_over_translation">忠实度</h3>
忠实度，即“信达雅”中的信，是翻译最起码的要求。不忠实主要表现为漏翻译和过翻译。
过度翻译指一些词或短语被重复地翻译，翻译不充分指部分词或短语没有被完整地翻译。该问题在神经机器翻译中普遍存在，包括基于注意力的神经机器翻译。

上述问题部分原因在于神经机器翻译并没有很好的机制来记忆历史翻译信息，比如已翻译词语信息和未翻译词语信息，从公式 13-15 可以看出。在这方面研究中，Tu 等人[23]提出的覆盖（Coverage）机制是很重要的研究成果。该方法将统计机器翻译
 

的覆盖机制引入基于注意力神经机器翻译。设计了一种覆盖向量，用于记录翻译过程的历史注意力信息，能够使注意力机制更多地关注未翻译词语，并降低已翻译词语权重。覆盖机制是统计机器翻译常用的方法，用于保证翻译的完整性。在神经机器翻译中，直接对覆盖机制建模是很困难的，Tu 等人通过在源语言编码状态中增加覆盖向量，显式地指导注意力机制的覆盖度。这种方法可以缓解过度翻译和翻译不充分问题，效果很明显。虽然没有完全解决该问题，但仍然是对注意力机制的重大改进。

该问题的另外一种解决方法是在翻译过程中控制源语言信息和目标语言信息对翻译结果的影响比例。这种思想很直观，在翻译过程中源语言上下文和目标语言上下文分别影响翻译忠实度和流利度。因此，当生成实词时应多关注源语言上下文，生成虚词时应更多依靠目标语言上下文。这就需要一种动态手段控制翻译过程中两种信息对翻译结果的影响，而这种控制手段是神经机器翻译所缺少的。这方面典型工作为 Tu 等人[47]提出的上下文门（Context Gate）方法，在保证翻译流利度同时，也确保了翻译的忠实度。覆盖机制和上下文门能够结合在一起，互为补充。覆盖机制能够生成更好的源语言上下文向量，着重考虑翻译充分性；上下文门则能够根据源语言、目标语言上下文的重要程度，动态控制两种信息対生成目标语言词语的影响比重。

过度翻译和翻译不充分问题是神经机器翻译存在的问题之一，在商用神经机器翻译系统中仍然存在该问题，需要更加深入研究。

有学者通过在神经机器翻译中融合重构（Reconstruction）思想，提高翻译忠实度

<h3 id="low_resource">资源稀缺</h3>

[资源稀缺](#low_resource)

<h4 id="data_argument">数据增强</h4>

<h4 id="monolingual_exploit">单语语料利用</h4>

<h4 id="pretraining">预训练模型</h4>

<h4 id="domain_transfer">跨领域迁移</h4>

<h4 id="multi_lingual_transfer">跨语言迁移</h4>

<h3 id="knowledge_merge">语言学知识融合</h3>

<h3 id="understanding">模型可解释性</h3>

<div align="center"><img src="https://github.com/lizezhonglaile/mt_tutorial/blob/main/pic/black-box.jpg" width="55%" height="55%"></div>
<div align="center">黑盒模型</div>

<h3 id="elegent_nmt">情感和文学性</h3>

<h2 id="evaluation">效果评测</h2>

<h3 id="public_eval">公开数据集和评测</h3>

<h3 id="human_eval">人工评测</h3>

<h3 id="auto_eval">自动评测</h3>

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

