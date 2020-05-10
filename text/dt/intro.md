Decision trees are an intuitive machine learning method that are naturally suited to a number of common tasks, particularly classification. At its most fundamental, a decision tree functions similarly to the game [Twenty Questions](https://en.wikipedia.org/wiki/Twenty_Questions). We interrogate a sample, with the answers to previous questions informing which question we ask next. Once we have asked enough questions to be confident in our answer, we predict what class the sample belongs to. 



You can think of this like starting at the trunk of an enormous tree, where each leaf represents an class. Each question represents a split in the branches, with the answer telling us which branch to follow in order to get to our desired leaf. The true power of a decision tree is being able to automatically generate these conditions to achieve accurate classifications. 

<img src="https://images.pexels.com/photos/38136/pexels-photo-38136.jpeg?auto=compress&amp;cs=tinysrgb&amp;dpr=2&amp;h=750&amp;w=1260" alt="Tree" width=100% />


## Growing a Tree

Let's consider the problem of classifying animals. This is a task decision trees are particularly suited for - one of the great triumphs of taxonomy has been classifying living organisms with the decision tree known as the [Tree of Life](https://en.wikipedia.org/wiki/Tree_of_life_(biology)). If we were tasked with creating a decision tree for this task, there are a number of sensible first questions we might ask. Is the animal warm or cold blooded? Do they have a spine? Do they lay eggs? Additionally, it is immediately obvious that our first question should not be "Can they rotate their hind flippers under their body?". While this might help distinguish between seals and sea lions, it provides very little information about animals in general.

Here you've demonstrated an intuitive understanding of a concept known as *Information Gain*. Although it was clear to us which features are more informative for this task, computers don't possess this intuition. Decision Trees need a way to algorithmically and automatically select the best feature to split by. In general, one of two metrics are used to determine feature importance: *Entropy* or *Gini*. Both of these provide a means of quantifying how much information is obtained by splitting a dataset by a given attribute. In general decision trees are built by greedily following these steps:

1.  Select the feature with the largest information gain
2. Split the data according to this feature
3. Repeat steps 1 and 2 for each split until each split contains only a single class.

Because we can keep splitting the groups finer and finer, Decision Trees are one of the few classification algorithms that (if not modified), will always achieve perfect accuracy on any training data.

## Overfitting and Pruning

> “Arrakis teaches the attitude of the knife - chopping off what's incomplete and saying: 'Now, it's complete because it's ended here.'
>
> \- Frank Herbert, *Dune*

While 100% accuracy on any dataset might sound excellent, it is actually a double edged sword. All machine learning methods suffer from what is known as the [Biance-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias–variance_tradeoff). Briefly, bias represents a model's inability to closely mirror the data, while variance represents the difference in performance between datasets. As Decision Trees can perfectly classify any training set they have effectively 0 bias, however this means that they rarely generalise well to unseen data i.e. they are high variance. This phenomenon is known as *overfitting*, and suggests the tree is modelling the noise and quirks of the training data, rather than the underlying distribution of the whole population. 

To combat this we implement *pruning*. As the name suggests, pruning involves removing branches from the tree that are causing the tree to become to specific to the training data. There are two main approaches to pruning, pre-pruning and post-pruning. 

### Pre-Pruning

Pre-pruning involves stopping the growth of the tree as it is being trained. This can be done in a number of ways, with 3 in particular discussed here. These are:

- Limiting the maximum depth of the tree
- Requiring a minimum improvement in *Gini* or *Information Gain* after a new split
- Requiring a minimum number of elements in the dataset to perform an additional split.

Limiting the maximum depth simply prevents a tree from branching too many times before a prediction is made. This prevents the tree from learning dataset specific splits, thus reducing overfitting. Requiring a minimum improvement in the selected metric at each split prevents splits occurring when such a split would only improve performance a tiny amount. Such a split is usually a sign of fitting to noise, and on an unseen dataset is likely to hinder performance. Finally, raising the minimum number of samples per split means that the tree will not generate splits finer and finer until perfect classification is achieved. 



### Post-Pruning

As the name suggests, post-pruning involves fully growing the decision tree, and then removing branches based on some criteria. There are multiple means of measuring which sections to prune, such as Reduced Error Pruning, Error Complexity Pruning, and Minimum Error Pruning. Whatever metric is selected, the basic premise follows the following process.

1. Evaluate each branch of the tree based on your chosen metric
2. Remove the branch that gives the greatest improvement of your metric
3. Repeat steps 1 and 2 until performance on a validation set stops improving.

For example, the simplest method Reduced Error Pruning simply evaluates the fully trained tree on a validation set. It then prunes the tree by removing the branches which give the biggest improvement to performance on the validation set. Once improvement slows down or stops, the pruning process is complete. 



Both pre and post pruning increase the bias of the decision tree slightly (reducing its performance on the training data), but decrease the variance (improve performance on unseen/test data). This leads to a more balanced model that will generalise better to unseen datasets and real world usage. In general, pre-pruning is more popular than post-pruning, as it can be done during the training process, and does not require an additional validation dataset. 



