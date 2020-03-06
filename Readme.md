**Summary**    
A skip-gram based implementation of the famous word2vec algorithm for generating text embeddings.
The two major portions of work are:     
i) word2vec using cross entropy loss and noise contrastive estimation(NCE) loss.    
ii) Evaluation of the implementation on wor analogy tasks viz. **King + Man - Woman ---> Queen**

**Scripts:**   
i) word2vec_basic.py: This file is the main script for training word2vec model.   
ii) loss_func.py:  This file has the two loss functions cross entropy and nce.   
iii) word_analogy.py: This file is for evaluating relation between pairs of words -- called MaxDiff question.  

**Startup:**    
word2vec_basic.py [cross_entropy | nce]

**References:**     
https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#cross-entropy
https://www.cs.toronto.edu/~amnih/papers/wordreps.pdf      
https://www.youtube.com/watch?v=kEMJRjEdNzM&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=2  
https://en.wikipedia.org/wiki/MaxDiff


