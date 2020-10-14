# 机器翻译技术调研
This is a machine translation reading list maintained by the Tsinghua Natural Language Processing Group. 
* [历史回顾](#history)
* [主要技术范式](#rule_machine_translation)
    * [基于规则的机器翻译](#rmt)
    * [基于实例的机器翻译](#ebmt)
    * [基于统计的机器翻译](#statistical_machine_translation)
    * [基于神经网络的机器翻译](#neural_machine_translation)
 * [效果评估](#evaluation)
 * [基于神经网络的机器翻译](#neural_machine_translation)
    * [Model Architecture](#model_architecture)
    * [Attention Mechanism](#attention_mechanism)
    * [Decoding](#decoding)
    * [Low-resource Language Translation](#low_resource_language_translation)
        * [Semi-supervised Learning](#semi_supervised)
        * [Unsupervised Learning](#unsupervised)
        * [Pivot-based Methods](#pivot_based)
 * [参考文献](#neural_machine_translation)

<h2 id="statistical_machine_translation">Statistical Machine Translation</h2>

<h3 id="smt_tutorials">Tutorials</h3>

* Philipp Koehn. 2006. [Statistical Machine Translation: the Basic, the Novel, and the Speculative](http://homepages.inf.ed.ac.uk/pkoehn/publications/tutorial2006.pdf). *EACL 2006 Tutorial*. ([Citation](https://scholar.google.com.hk/scholar?cites=226053141145183075&as_sdt=2005&sciodt=0,5&hl=en): 10)
* Adam Lopez. 2008. [Statistical Machine Translation](http://delivery.acm.org/10.1145/1390000/1380586/a8-lopez.pdf?ip=101.5.129.50&id=1380586&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E587F3204F5B62A59%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1546058891_981e84a24804f2dbc0549b9892a2ea1d). *ACM Computing Surveys*. ([Citation](https://scholar.google.com.hk/scholar?cites=13327711981648149476&as_sdt=2005&sciodt=0,5&hl=en): 373)

<h3 id="word_based_models">Word-based Models</h3>

* Peter E. Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, and Robert L. Mercer. 1993. [The Mathematics of Statistical Machine Translation: Parameter Estimation](http://aclweb.org/anthology/J93-2003). *Computational Linguistics*. ([Citation](https://scholar.google.com/scholar?cites=2259057253133260714&as_sdt=2005&sciodt=0,5&hl=en): 4,965)
* Stephan Vogel, Hermann Ney, and Christoph Tillmann. 1996. [HMM-Based Word Alignment in Statistical Translation](http://aclweb.org/anthology/C96-2141). In *Proceedings of COLING 1996*. ([Citation](https://scholar.google.com.hk/scholar?cites=6742027174667056165&as_sdt=2005&sciodt=0,5&hl=en): 940)
* Franz Josef Och and Hermann Ney. 2003. [A Systematic Comparison of Various Statistical Alignment Models](http://aclweb.org/anthology/J03-1002). *Computational Linguistics*. ([Citation](https://scholar.google.com.hk/scholar?cites=7906670690027479083&as_sdt=2005&sciodt=0,5&hl=en): 3,980)
* Percy Liang, Ben Taskar, and Dan Klein. 2006. [Alignment by Agreement](https://cs.stanford.edu/~pliang/papers/alignment-naacl2006.pdf). In *Proceedings of NAACL 2006*. ([Citation](https://scholar.google.com.hk/scholar?cites=10766838746666771394&as_sdt=2005&sciodt=0,5&hl=en): 452)
* Chris Dyer, Victor Chahuneau, and Noah A. Smith. 2013. [A Simple, Fast, and Effective Reparameterization of IBM Model 2](http://www.aclweb.org/anthology/N13-1073). In *Proceedings of NAACL 2013*. ([Citation](https://scholar.google.com.hk/scholar?cites=13560076980956479370&as_sdt=2005&sciodt=0,5&hl=en): 310)

<h3 id="phrase_based_models">Phrase-based Models</h3>

* Philipp Koehn, Franz J. Och, and Daniel Marcu. 2003. [Statistical Phrase-Based Translation](http://aclweb.org/anthology/N03-1017). In *Proceedings of NAACL 2003*. ([Citation](https://scholar.google.com.hk/scholar?cites=11796378766060939113&as_sdt=2005&sciodt=0,5&hl=en): 3,516)

<h2 id="evaluation">Evaluation</h2>

* Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://aclweb.org/anthology/P02-1040). In *Proceedings of ACL 2002*. ([Citation](https://scholar.google.com.hk/scholar?cites=9019091454858686906&as_sdt=2005&sciodt=0,5&hl=en): 8,499)
* Philipp Koehn. 2004. [Statistical Significance Tests for Machine Translation Evaluation](http://www.aclweb.org/anthology/W04-3250). In *Proceedings of EMNLP 2004*. ([Citation](https://scholar.google.com.hk/scholar?cites=6141850486206753388&as_sdt=2005&sciodt=0,5&hl=en): 1,015)
* Satanjeev Banerjee and Alon Lavie. 2005. [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](http://aclweb.org/anthology/W05-0909). In *Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*. ([Citation](https://scholar.google.com.hk/scholar?cites=11797833340491598355&as_sdt=2005&sciodt=0,5&hl=en): 1,355)

<h2 id="neural_machine_translation">Neural Machine Translation</h2>

<h3 id="nmt_tutorials">Tutorials</h3>

* Thang Luong, Kyunghyun Cho, and Christopher Manning. 2016. [Neural Machine Translation](https://nlp.stanford.edu/projects/nmt/Luong-Cho-Manning-NMT-ACL2016-v4.pdf). *ACL 2016 Tutorial*.  

<h2 id="neural_machine_translation">参考文献</h2>

* Graham Neubig. 2017. [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/pdf/1703.01619.pdf). *arXiv:1703.01619*. ([Citation](https://scholar.google.com/scholar?cites=17621873290135947085&as_sdt=2005&sciodt=0,5&hl=en): 45)
* Oriol Vinyals and Navdeep Jaitly. 2017. [Seq2Seq ICML Tutorial](https://docs.google.com/presentation/d/1quIMxEEPEf5EkRHc2USQaoJRC4QNX6_KomdZTBMBWjk/present?slide=id.p). *ICML 2017 Tutorial*.
