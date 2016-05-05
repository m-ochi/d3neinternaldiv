### D3NE(Domain Dependency Decomposition Network Embedding) method (Internally dividing model)
--------------------------------

##### Topic
- D3NE method with internally dividing version.

##### Introduction
- This method uses three network graphs.
- First, the word-word cooccurence network. This is made from cooccurence probability in all text windows.
- Second, the document-word cooccurence network. This is made from word occurence probability in each document.
- Third, the label-word coocurence netwrok. This is made from word distribution probability in each label.
- **This method suppose that the word-word network is generated from the word-label and word-document networks.**
- This method can embed each node to a vector space with some dimensions.
- Nodes are words, documents, or labels, and they are embedded in the same vector space.
- This method is categolized to the semi-supervised learning style, so it is not necessary to be labeled for all words. If a part of words are labeled, then this method can be embedded correctly.

- I wrote this code based on the LINE code. [LINE](https://github.com/tangjianpku/LINE)

##### Directory structure
- dataminimal/: Just sample data for test run.
- src/: The source codes and the shell script for test compile and run.
- tools/: Visualization codes (TSNE method) or other toys.

##### How to compile and run the sample data
- This code uses `CMake`

```
cd src/
./run.sh
```

- The `make` command generates the compiled file `train_pte`, this is main file to run the D3NE method. 

##### Features
- This code use **C++11** functions.
- Speeding up sampling using the **Alias table method**
- Using ASGD(Asyncronized Stochastic Gradient Descent) method based on the **HogWild!** paper.

##### train_pte options
-  --train_ww=./data/st-st.csv : specify the word-word network file
-  --train_wd=./data/st-corp.csv： specify the word-document network file
-  --train_wl=./data/st-imp.csv：specify the word-label network file
-  --output=vec_2nd_wo_norm.txt：specif the output vector file
-  --binary=0：binary format output (don't use it because of not debugging now)
-  --size=128：specify the vector dimension size.
-  --order=2： specify 1 or 2, this means specifying the first or second proximity, detail information is written in the original paper.
-  --negative=5：specify the iteration number of negative sampling for each sampling.
-  --samples=30：specify the total number of sampling. (1 means 1\*10^6 samplings)
-  --threads=30：specify the number of using CPUs.
-  --rho=0.05：specify the initial learning rate. (this rate reduce gradually while running)

```./train_d3ne --help``` shows the detail option information

##### Output files
- vec_2nd_wo_norm.txt_all: This is the internally dividing model file, so each node vector values are the sum of *wd file and *wl file values.
- vec_2nd_wo_norm.txt_wd: This is the word-document vector file.
- vec_2nd_wo_norm.txt_wl: This is the word-label vector file.

##### Output file format
- First row: 'total number of vertex' 'vectors dimension size'
- After second row: 'the name of vertex' each dimension value with learned vector
- All characters are separated with 'space'.

##### Developmental status
- First version was released(03/2016)
- In this version, I implemented only the `Joint Training` style.
    - Seaquentially update all three graphs in each learning loop.
