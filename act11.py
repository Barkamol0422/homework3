from transformers import pipeline

classifier=pipeline("sentiment-analysis")
a=str(input("Entera text: "))
result=classifier(a)
print(result[0]["label"], result[0]["score"])
