# Code of Contrastive Learning for Image Captioning

Based on [Neuraltalk2](https://github.com/karpathy/neuraltalk2) and [AdaptiveAttention](https://github.com/jiasenlu/AdaptiveAttention). Special thanks to the authors!

Testing codes and related codes will be added gradually.

When training using contrastive learning, better pre-train both the target and the reference.

## Briefy Explanation 

- adaptive attention: model structure for adaptive attention
- neuraltalk2: model structure for neuraltalk2
- misc: codes for DataLoader, util functions, and codes related to contrastive learning
	- cl_adaptiveattention.lua: codes for using adaptiveattention as both target model and reference model
	- cl_cross.lua: codes for using adaptiveattention as reference model, and using neuraltalk2 as target model
- cl_train_adaptiveattention.lua: codes for training adaptiveattention
- cl_train_cross.lua: codes for training neuraltalk2, using adaptiveattention as reference model



