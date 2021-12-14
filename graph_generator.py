import matplotlib.pyplot as plt
import pandas as pd

# load analysis file into analysis
analysis = pd.read_csv("analysis.csv")
model_name = analysis["model name"]
accuracy = analysis["accuracy"]

# array for accuracy in 100
output_accuracy = []
for x in range(0,len(accuracy)):
    output_accuracy.append(accuracy[x] * 100)


# ploting
plt.bar(model_name,output_accuracy)
plt.xticks(rotation = 90)
plt.title("Accuracy of model")
plt.savefig('accuracy-distribution.pdf')