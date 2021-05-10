* [(四) 神经机器翻译前沿：挑战和现状](#challenges)
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


<h2 id="challenges">(四) 神经机器翻译前沿：挑战和现状</h2>

<h3 id="oov">受限词表</h3>

<h4 id="subword_nmt">子词模型</h4>

R. Sennrich, B. Haddow, A. Birch, Neural machine translation of rare words with subword units, in: Proceedings of ACL, 2016.

<h4 id="char_nmt">字符模型</h4>

J. Lee, K. Cho, T. Hofmann, Fully character-level neural machine translation without explicit segmentation, Transactions of the Association for Computational Linguistics 5 (2017) 365–378.


<h4 id="hybrid_nmt">混合模型</h4>

[106] M.-T. Luong, C. D. Manning, Achieving open vocabulary neural machine translation with hybrid word-character models, arXiv preprint arXiv:1604.00788 (2016).

<h4 id="byte_nmt">字节模型</h4>

C. Wang, K. Cho, J. Gu, Neural machine translation with byte-level subwords, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 2020, pp. 9154–9160.

字符级神经机器翻译（Character Level NMT）是为了解决未登录词、词语切分、词语形态变化等问题提出的一种神经机器翻译模型，主要特点是减小了输入和输出粒度。
词语编码方案

多数神经机器翻译模型都以词语作为翻译基本单位，存在未登录词、数据稀疏，以及汉语、日语等语言中的分词问题。此外，在形态变化较多的语言中，如英语、法语等语言，以词为处理基本单位时，丢失了词语之间的形态变化、语义信息。如英语单词，“run”，“runs”，“ran”，“running”被认为是四个不同的词，忽略了他们有着共同的前缀“run”。为了解决上述问题，学者们提出了不同的词语编码方案，根据粒度划分可以归为以下两种：

（1）字符编码方案。对于英语、法语等拼音文字来说字符是组成词语的基本单位，在语言处理中能够以字符为单位建模。这方面工作很早就开始研究，比如字符级神经网络语言模型[48]。该方案同时也存在不足，比如编码粒度过小，适合英语、法语等字符数量相近的语言之间的翻译，如果用在英语到汉语翻译上会出现诸多问题。

（2）亚词编码方案。亚词编码方案选用的翻译基本单位介于字符和词语，可以得到两种方案的共同优势。词素的粒度同样介于字符和词语之间，不足之处是跟特定语言相关，限制了应用的通用性。因此，亚词通常采用 BPE 编码（Byte Pair Encoding, BPE）得到[49]，该方案将经常结合的字符组合看作是一个单位，比如词语 “ dreamworks interactive”，可以切分成“dre + am + wo + rks/ in + te + ra + cti + ve”序列，方法简单有效，适应性强。

基于字符的翻译模型


<h3 id="loyal_translation">忠实度</h3>

Z. Tu, Z. Lu, Y. Liu, X. Liu, H. Li, Modeling coverage for neural machine translation, in: Proceedings of ACL, 2016.

忠实度，即“信达雅”中的信，是翻译最起码的要求。不忠实主要表现为漏翻译和过翻译。
过度翻译指一些词或短语被重复地翻译，翻译不充分指部分词或短语没有被完整地翻译。该问题在神经机器翻译中普遍存在，包括基于注意力的神经机器翻译。

上述问题部分原因在于神经机器翻译并没有很好的机制来记忆历史翻译信息，比如已翻译词语信息和未翻译词语信息，从公式 13-15 可以看出。在这方面研究中，Tu 等人[23]提出的覆盖（Coverage）机制是很重要的研究成果。该方法将统计机器翻译
 

的覆盖机制引入基于注意力神经机器翻译。设计了一种覆盖向量，用于记录翻译过程的历史注意力信息，能够使注意力机制更多地关注未翻译词语，并降低已翻译词语权重。覆盖机制是统计机器翻译常用的方法，用于保证翻译的完整性。在神经机器翻译中，直接对覆盖机制建模是很困难的，Tu 等人通过在源语言编码状态中增加覆盖向量，显式地指导注意力机制的覆盖度。这种方法可以缓解过度翻译和翻译不充分问题，效果很明显。虽然没有完全解决该问题，但仍然是对注意力机制的重大改进。

该问题的另外一种解决方法是在翻译过程中控制源语言信息和目标语言信息对翻译结果的影响比例。这种思想很直观，在翻译过程中源语言上下文和目标语言上下文分别影响翻译忠实度和流利度。因此，当生成实词时应多关注源语言上下文，生成虚词时应更多依靠目标语言上下文。这就需要一种动态手段控制翻译过程中两种信息对翻译结果的影响，而这种控制手段是神经机器翻译所缺少的。这方面典型工作为 Tu 等人[47]提出的上下文门（Context Gate）方法，在保证翻译流利度同时，也确保了翻译的忠实度。覆盖机制和上下文门能够结合在一起，互为补充。覆盖机制能够生成更好的源语言上下文向量，着重考虑翻译充分性；上下文门则能够根据源语言、目标语言上下文的重要程度，动态控制两种信息対生成目标语言词语的影响比重。

过度翻译和翻译不充分问题是神经机器翻译存在的问题之一，在商用神经机器翻译系统中仍然存在该问题，需要更加深入研究。

有学者通过在神经机器翻译中融合重构（Reconstruction）思想，提高翻译忠实度

<h3 id="low_resource">资源稀缺</h3>

<h4 id="data_argument">数据增强</h4>

<h4 id="monolingual_exploit">单语语料利用</h4>

R. Sennrich, B. Haddow, A. Birch, Improving neural machine translation models with monolingual data, in: Proceedings of ACL, 2016, pp.86–96.

V. C. D. Hoang, P. Koehn, G. Haffari, T. Cohn, Iterative back-translation for neural machine translation, in: Proceedings of the 2nd Workshop on Neural Machine Translation and Generation, 2018, pp. 18–24.

Y. Cheng,W. Xu, Z. He,W. He, H.Wu, M. Sun, Y. Liu, Semi-supervised learning for neural machine translation, in: Proceedings of ACL, 2016, pp. 1965–1974.

[78] D. He, Y. Xia, T. Qin, L. Wang, N. Yu, T.-Y. Liu, W.-Y. Ma, Dual learning for machine translation, in: Advances in NeurIPS, 2016, pp. 820–828.

单语语料是一种非常重要的资源，具有数量大、获取方便的优势。在统计机器翻译中，大规模目标语言单语语料可以提供优质的语言模型，对提高翻译流利度起着很重要作用。在神经机器翻译中可以利用的单语语料主要分为目标语言单语语料和源语言单语语料。

目标语言单语语料应用之一是语言模型， Gulcehre 等人提出一种利用大规模单语语料提高神经机器翻译效果的方法。采用单语语料训练神经网络语言模型，将之集成到神经机器翻译中，集成方法分为浅层集成和深层集成。浅层集成方法在解码时，把语言模型作为一种特征用来生成候选词；深层集成方法将神经机器翻译模型、语言模型的隐藏状态连接在一起，通过控制机制动态平衡两种模型对解码的影响，在解码时可以捕捉到语言模型信息。这两种集成方法均可以提高翻译效果，其中深层集成方法效果更为明显 。 此外， Domhan 等人则提出采用多任务学习方法，将神经机器翻译模型和目标语言的语言模型联合训练，以此利用大规模目标语言单语语料。

目标语言单语语料的另一使用方法是 Sennrich 等人提出的训练数据构造方法：回翻译（Back-translation）方法。利用目标语言单语语料构造伪双语数据，并加入到训练语料。这种融合方法对神经机器翻译模型不作改变，方法简单有效，虽然在一定程度上提高了翻译效果，但是效果提升取决于构造数据的质量。

以上研究都是利用目标语言单语语料，Zhang 等人提出了将源语言单语语料应用到神经机器翻译的方法。实现方式有两种，第一种方法同样采用了构造数据思想，在构造方式上通过自学习的方法扩大双语训练语料规模；另外一种方法通过多任务学习增强编码器对源语言的表示质量。这两种方法均能够大幅提升翻译效果。不足之处是源语言单语语料的数量和题材会对翻译模型性能产生影响。

同时利用源语言和目标语言单语语料，主要有Cheng 等人提出的半监督学习方法。基本思想是将自编码引入源语言到目标语言翻译模型和目标语言到源语言翻译模型，通过半监督方法训练双向神经机器翻译，以此利用源语言和目标语言单语语料提高翻译效果。这种方法显著优势是可以同时利用源语言和目标语言的单语语料，不足之处是对单语语料中的未登录词没有处理能力。
此外，Ramachandran 等人提出一种更为简单的方法，将序列到序列模型看作为两个语言模型，通过大规模单语语料分别训练源语言和目标语言的语言模型；神经机器翻译模型的编码器和解码器参数分别由两个语言模型参数初始化；然后利用双语平行语料训练，训练过程中语言模型参数同时调整。

<h4 id="pretraining">预训练模型</h4>

M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, L. Zettlemoyer, Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, arXiv preprint arXiv:1910.13461 (2019).

S. Edunov, A. Baevski, M. Auli, Pre-trained language model representations
for language generation, arXiv preprint arXiv:1903.09722 (2019).

J. Zhu, Y. Xia, L. Wu, D. He, T. Qin, W. Zhou, H. Li, T.-Y. Liu, Incorporating bert into neural machine translation, arXiv preprint
arXiv:2002.06823 (2020).

<h4 id="domain_transfer">跨领域迁移</h4>

<h4 id="multi_lingual_transfer">跨语言迁移</h4>

<h3 id="knowledge_merge">知识融合</h3>

<h4 id="word_knowledge_merge">词汇知识</h4>
外部词典
词性多特征嵌入
词对齐


133 Xing Wang, Zhaopeng Tu, and Min Zhang. Incorporating statistical machine translation word knowledge into neural machine translation. IEEE/ACM Transactions on Audio, Speech,and Language Processing, 26(12):2255–2266, 2018.

R. Sennrich, B. Haddow, Linguistic input features improve neural machine translation, in: Proceedings of WMT, 2016, pp. 83–91.

<h4 id="phrase_knowledge_merge">短语知识</h4>
外部词典
短语编码和解码

Xing Wang, Zhaopeng Tu, Deyi Xiong, and Min Zhang. Translating phrases in neural machine translation. In Proceedings of EMNLP, 2017.


<h4 id="syntax_merge">句法知识</h4>

源端句法知识

A. Eriguchi, K. Hashimoto, Y. Tsuruoka, Tree-to-sequence attentional neural machine translation, in: Proceedings of ACL, 2016, pp. 823–833.

J. Hao, X.Wang, S. Shi, J. Zhang, Z. Tu, Multi-granularity self-attention for neural machine translation, in: Proceedings of EMNLP-IJCNLP, 2019, pp. 886–896.

E. Bugliarello, N. Okazaki, Enhancing machine translation with dependency-aware self-attention, in: Proceedings of ACL, 2020, pp. 1618–1627.

J. Bastings, I. Titov, W. Aziz, D. Marcheggiani, K. Sima’an, Graph convolutional encoders for syntax-aware neural machine translation, in: Proceedings of EMNLP, 2017, pp. 1957–1967.

目标端句法知识

A. Eriguchi, Y. Tsuruoka, K. Cho, Learning to parse and translate improves neural machine translation, in: Proceedings of ACL, 2017, pp. 72–78.

J. G¯u, H. S. Shavarani, A. Sarkar, Top-down tree structured decoding with syntactic connections for neural machine translation and parsing, in: Proceedings of EMNLP, 2018, pp. 401–413.

X. Wang, H. Pham, P. Yin, G. Neubig, A tree-based decoder for neural machine translation, in: Proceedings of EMNLP, 2018, pp. 4772–4777.

S. Wu, D. Zhang, N. Yang, M. Li, M. Zhou, Sequence-to-dependency neural machine translation, in: Proceedings of ACL, 2017, pp. 698–707.

R. Aharoni, Y. Goldberg, Towards string-to-tree neural machine translation, in: Proceedings of ACL, 2017, pp. 132–140.

J. Yang, S. Ma, D. Zhang, Z. Li, M. Zhou, Improving neural machine translation with soft template prediction, in: Proceedings of WMT, 2020, pp. 5979–5989.

<h4 id="world_knowledge_merge">世界知识</h4>

<h3 id="decoding_space">解码问题</h3>

<h4 id="decoding_bi">双向解码</h4>

<h4 id="decoding_heir">层次解码</h4>

<h4 id="decoding_parallel">非自回归解码</h4>

Jiatao Gu, James Bradbury, Caiming Xiong, Victor OK Li, and Richard Socher. Non-autoregressive neural machine translation. In Proceedings of ICLR, 2018.

<h4 id="decoding_mutli_pass">二次解码</h4>

<h3 id="understanding">模型可解释性</h3>

<h4 id="explict_model">显式建模</h4>

F. Stahlberg, D. Saunders, B. Byrne, An operation sequence model for explainable neural machine translation, in: Proceedings of EMNLP Workshop, 2018, pp. 175–186.

<h4 id="visualize">可视化</h4>

Y. Ding, Y. Liu, H. Luan, M. Sun, Visualizing and understanding neural machine translation, in: Proceedings of ACL, 2017, pp. 1150–1159.

H. Strobelt, S. Gehrmann, M. Behrisch, A. Perer, H. Pfister, A. M. Rush, Seq2seq-vis: A visual debugging tool for sequence-to-sequence models, IEEE transactions on visualization and computer graphics 25 (2019) 353–363.

S. He, Z. Tu, X.Wang, L.Wang, M. Lyu, S. Shi, Towards understanding neural machine translation with word importance, in: Proceedings of EMNLP-IJCNLP, 2019, pp. 953–962.

A. Raganato, J. Tiedemann, An analysis of encoder representations in transformer-based machine translation, in: Proceedings of EMNLP Workshop, 2018, pp. 287–297.


<h3 id="model_consistence">模型一致性</h3>

M. Ranzato, S. Chopra, M. Auli, W. Zaremba, Sequence level training with recurrent neural networks, in: Proceedings of ICLR, 2016.

<h3 id="model_control">模型可控性</h3>

<h3 id="model_robust">模型健壮性</h3>

Y. Cheng, Z. Tu, F. Meng, J. Zhai, Y. Liu, Towards robust neural machine translation, in: Proceedings of ACL, 2018, pp. 1756–1766.

<h3 id="elegent_nmt">情感和文学性</h3>
