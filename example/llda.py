import os

import model.LLDA as llda
from preprocess.preprocess import get_labeled_docs

# data
data_path = os.path.dirname(os.path.realpath(os.getcwd()))
data_path = os.path.join(data_path, 'data')

labeled_docs = get_labeled_docs(os.path.join(data_path, 'data_test', 'test_test'))

# test
labeled_documents = [("example example example example example", ["example"]),
                     ("test llda model test llda model test llda model", ["test", "llda_model"]),
                     ("example test example test example test example test", ["example", "test"]),
                     ("good perfect good good perfect good good perfect good ", ["positive"]),
                     ("bad bad down down bad bad down", ["negative"])]

llda_model = llda.LldaModel(labeled_documents=labeled_docs)
print(llda_model)

while True:
    print("iteration %s sampling..." % (llda_model.iteration + 1))
    llda_model.training(1)
    print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity))
    if llda_model.is_convergent:
        break

document = labeled_docs[1][0]
topics = llda_model.inference_multi_processors(document=document, iteration=10, times=10)
print(topics)
