import sys

TOOLS_PATH = "./VQAEvalTools/VQA-master"
sys.path.append(TOOLS_PATH + "/PythonEvaluationTools")
sys.path.append(TOOLS_PATH + "/PythonHelperTools")

from vqaEvaluation import vqaEval
from vqaTools import vqa

annFile = "./annotate.json"
quesFile = "./questions.json"
resFile = "./results.json"

load_vqa = vqa.VQA(annFile, quesFile)
vqaRes = load_vqa.loadRes(resFile, quesFile)

myvqaEval = vqaEval.VQAEval(load_vqa, vqaRes, n=2)

myvqaEval.evaluate()

print("OVERALL ACCURACY: " + str(myvqaEval.accuracy["overall"]))
print("Overall Accuracy is: %.02f\n" % (myvqaEval.accuracy["overall"]))
print("=====================")
print("Per Question Type Accuracy is the following:")
for quesType in myvqaEval.accuracy["perQuestionType"]:
    print(str(quesType) + " : " + str(myvqaEval.accuracy["perQuestionType"][quesType]))
print("=====================")
print("Per Answer Type Accuracy is the following:")
for ansType in myvqaEval.accuracy["perAnswerType"]:
    print(str(ansType) + " : " + str(myvqaEval.accuracy["perAnswerType"][ansType]))
