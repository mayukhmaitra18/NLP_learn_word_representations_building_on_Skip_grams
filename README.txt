Readme

Model link: https://drive.google.com/drive/folders/1uO2DFWYsnIb5GCsA53DUWkztOvzyk4Py?usp=sharing

Implementation details:
1. Completion of batch generation module in word2vec_basic.py


2. Completion of loss functions in loss_func.py
   1. Cross entropy: 
      1. Took the dot product of the context embedding and target embedding
      2. Removed the nan values from the resultant dot product generated above
      3. Took log of the dot product
      4. Then calculated the scores for each context and target word combinations
   2. NCE loss:
      1. From the given words, first extracted the target embeddings
      2. Then calculated the target words’ unigram probabilities
      3. Then calculated the target words’ bias
      4. Took the dot product of the context embedding and target embedding
      5. Calculated log(kP(x))
      6. Calculated P(D=1 Wo|Wc)
      7. Repeated step i to iv for negative words
      8. Calculated log(kP(wx))
      9. Completed the final equation
      10. Also, to prevent nan values during training added very small correction values to the tensors (to be precise: 0.0000000001)


3. Configurations tested for:
Hyperparameter configuration
	batch_size 128, embedding_size 128, skip_window 4, num_skips 8,	num_sampled 64,	max_num_steps 200001
	batch_size 100, embedding_size 128, skip_window 4, num_skips 5,	num_sampled 64,	max_num_steps 200001
	batch_size 120, embedding_size 128, skip_window 5, num_skips 10,num_sampled 64,	max_num_steps 300001
	batch_size 128, embedding_size 128, skip_window 1, num_skips 2,	num_sampled 32,	max_num_steps 100001
	batch_size 256, embedding_size 128, skip_window 4, num_skips 8,	num_sampled 64,	max_num_steps 200001
	batch_size 512, embedding_size 128, skip_window 4, num_skips 8,	num_sampled 64,	max_num_steps 100001
	
The other configurations remained unchanged: valid_size = 16, valid_window = 100, checkpoint step = 50,000


4. Training steps: Training was first started using the pretrained model, but was later changed and made to resume from my checkpoints, which helped in increasing accuracy.
   1. New models were generated for both cross entropy and nce loss functions with the above configurations and analogy was done using all those models separately.
5. Analogy: completed word_analogy.py to generate prediction files. Used cosine similarity scores to get context words using both NCE and cross entropy loss models


6. Execution steps:
   1. python word2vec_basic.py (run this command to train model using cross entropy loss function)
   2. Python word2vec_basic.py nce (run this command to train model using nce loss function)
   3. Once word_analogy.py code is completed, set loss_model = ‘cross_entropy’ and comment out #loss_model = ‘cross_entropy’ to run analogy for cross_entropy model.
   4. Also, in word_analogy.py, change the output file name accordingly when executing for cross entropy or nce models
   5. Python word_analogy.py (run this command after making the above changes according to your needs to get analogy results)
   6. For running analogy for the test file change filename at statement with open('word_analogy_test.txt') as f:
   7. To get accuracy scores on your predictions run the following command:
      1.  ./score_maxdiff.pl word_analogy_dev_mturk_answers.txt <prediction_file_name> <output_file_name>