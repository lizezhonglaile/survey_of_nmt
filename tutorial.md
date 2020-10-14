# 机器翻译技术调研
* [背景介绍](#history)
* [主要技术范式](#rule_machine_translation)
    * [直接机器翻译](#rmt)
    * [句法转换机器翻译](#rmt)
    * [基于中间语言的机器翻译](#rmt)
    * [基于规则的机器翻译](#rmt)
    * [基于实例的机器翻译](#ebmt)
    * [基于统计的机器翻译](#statistical_machine_translation)
    * [基于神经网络的机器翻译](#neural_machine_translation)
* [神经机器翻译](#neural_machine_translation)
    * [第一个seq2seq模型](#model_architecture)
    * [注意力模型](#model_architecture)
    * [transformer模型](#model_architecture)
    * [短语神经模型](#model_architecture)
    * [融合句法的神经模型](#model_architecture)
    * [多语言](#model_architecture)
    * [多模态](#model_architecture)
    * [非平行语料利用](#model_architecture)
    * [解码](#model_architecture)
    * [诗歌翻译](#low_resource_language_translation)
* [挑战和进展](#neural_machine_translation)
    * [稀疏词问题](#model_architecture)
    * [漏翻译和过翻译](#attention_mechanism)
    * [低资源问题](#decoding)
    * [知识融合](#decoding)
    * [跨领域跨语言和跨模态的迁移学习](#low_resource_language_translation)
    * [模型可解释性](#low_resource_language_translation)
* [效果评估](#evaluation)
    * [人工评测](#model_architecture)
    * [自动评测](#model_architecture)
    * [公开评测](#model_architecture)
* [参考文献](#neural_machine_translation)

<h2 id="statistical_machine_translation">背景介绍</h2>

<h2 id="statistical_machine_translation">主要技术范式</h2>

<h3 id="smt_tutorials">直接机器翻译</h3>

<h3 id="smt_tutorials">句法转换机器翻译</h3>

<h3 id="smt_tutorials">基于中间语言的机器翻译</h3>

<h3 id="smt_tutorials">基于规则的机器翻译</h3>

<h3 id="smt_tutorials">基于实例的机器翻译</h3>

<h3 id="smt_tutorials">基于神经网络的机器翻译</h3>

<h2 id="neural_machine_translation">神经机器翻译</h2>

<h3 id="smt_tutorials">第一个seq2seq模型</h3>

<h3 id="smt_tutorials">注意力模型</h3>

<h3 id="smt_tutorials">transformer模型</h3>

<h3 id="smt_tutorials">短语神经模型</h3>

<h3 id="smt_tutorials">融合句法的神经模型</h3>

<h3 id="smt_tutorials">多语言</h3>

<h3 id="smt_tutorials">多模态</h3>

<h3 id="smt_tutorials">非平行语料利用</h3>

<h3 id="smt_tutorials">解码</h3>

<h3 id="smt_tutorials">诗歌翻译</h3>

<h2 id="neural_machine_translation">挑战和进展</h2>

<h3 id="smt_tutorials">稀疏词问题</h3>

<h3 id="smt_tutorials">漏翻译和过翻译</h3>

<h3 id="smt_tutorials">低资源问题</h3>

<h3 id="smt_tutorials">知识融合</h3>

<h3 id="smt_tutorials">跨领域跨语言和跨模态的迁移学习</h3>

<h3 id="smt_tutorials">模型可解释性</h3>

<h3 id="smt_tutorials">多模态</h3>

<h2 id="evaluation">效果评估</h2>

<h3 id="smt_tutorials">人工评测</h3>

<h3 id="smt_tutorials">自动评测</h3>

<h3 id="smt_tutorials">公开评测</h3>

<h2 id="neural_machine_translation">参考文献</h2>

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

<h2 id="neural_machine_translation">参考文献</h2>

* Graham Neubig. 2017. [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/pdf/1703.01619.pdf). *arXiv:1703.01619*. ([Citation](https://scholar.google.com/scholar?cites=17621873290135947085&as_sdt=2005&sciodt=0,5&hl=en): 45)
* Oriol Vinyals and Navdeep Jaitly. 2017. [Seq2Seq ICML Tutorial](https://docs.google.com/presentation/d/1quIMxEEPEf5EkRHc2USQaoJRC4QNX6_KomdZTBMBWjk/present?slide=id.p). *ICML 2017 Tutorial*.
