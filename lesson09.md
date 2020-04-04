## 1.思考在自然语言处理中如何发掘模型的可解释性
在进行自然语言处理时，如果是数据量小的情况，就可以选择一些传统机器学习模型，可解释性强，比如决策树模型、线性回归、逻辑回归、朴素贝叶斯模型等。而对于数据量庞大的深度学习模型来说，可以对隐层进行重构，生成原句的方式进行分析；使用激活最大化的方式，即对部分词组进行遮挡来确定模型选取的最大特征，并进行可视化操作；添加对抗数据，使用近似词替换和改变词组顺序的方式对模型进行解读。

## 2.在Seq2Seq和注意力机制中如何可视化模型细节
seq2seq和注意力机制的模型内部都是由 编码器(encoder) 和 解码器(decoder) 组成的，其权重可以作为模型解释的一个指标，可以将每层的权重与对应词组进行可视化分析。

## 3.对抗样本能否运用到自然语言处理模型中
对抗样本可以运用到自认语言处理模型之中，比如通过移除、添加或调序单词内部的字符，来构建更稳健的文本分类模型，通过这些扰动，模型能学会如何处理错别字，从而不至于对分类结果产生影响。

## 4.复现Kaggle心脏病数据集冠军kernel，理解所用的模型可解释性技巧
Kaggle上心脏病数据集冠军kernel使用的是随机森林模型，在加载数据，训练模型之后，作者首先绘制了顺向决策树，来查看模型做了什么，代码如下
```
export_graphviz(estimator,out_file='tree.dot',
               feature_names = feature_names,
               class_names = y_train_str,
               rounded = True, proportion = True,
               label='root',
               precision = 2,filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
from IPython.display import Image
Image(filename='tree.png')
```
决策树绘制图能表示子节点信息是如何拆分的，并不能直观的反映出预测时的显著特征。
作者在模型可解释性方面主要应用了Permutation Importance 、Partial Dependence Plots（PDP） 、SHAP Values三种方法，以达到洞察模型数据特征的目的，具体原理如下：

##### 1.Permutation Importance
Permutation Importance 方法就是就是在模型训练好之后，将某一特征数据随机打乱顺序，然后重新用模型预测，看看此时模型的metric或者loss function变化了多少，每个特征重复这样的操作，来比较打乱顺序后对模型预测影响最大的特征，代码如下：
```
perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())
```

##### 2.Partial Dependence Plots（PDP）
Partial Dependence Plots方法也是在模型训练好后，通过改变单一变量在一个值范围内时对输出（预测）结果的影响，在本例中，作者使用了'num_major_servers' 'age' 'st_depression'三个变量进行分析，代码如下：
```
#这是对'num_major_servers'变量的分析
base_features = dt.columns.values.tolist()
base_features.remove('target')#delete the target from the list
feat_name = 'num_major_vessels'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

#这是对'age'变量的分析
feat_name = 'age'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

#这是对'st_depression'变量的分析
feat_name = 'st_depression'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
```
此时，作者还应用了2D PDP方法来验证变量'st_depression'斜率对输出结果的影响
```
inter1  =  pdp.pdp_interact(model=model, dataset=X_test, model_features=base_features, features=['st_slope_upsloping', 'st_depression'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['st_slope_upsloping', 'st_depression'], plot_type='contour')
plt.show()

inter1  =  pdp.pdp_interact(model=model, dataset=X_test, model_features=base_features, features=['st_slope_flat', 'st_depression'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['st_slope_flat', 'st_depression'], plot_type='contour')
plt.show()
```
##### 3. SHAP Values
SHAP Values方法是对各个特征求得其对预测结果的影响，简单来说就是单个特征对结果的贡献值，计算原理就是某特征在特征行列中的权重乘以新增特征前后的变化值，通过对比单个变量与它们的基准值来观察对结果的影响。
```
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")
```
可以显示出各个特征对是否得病的影响。