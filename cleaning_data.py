from imports import *

reviews = pd.read_csv("amazon_dataset/Reviews.csv")
print (reviews.shape)
print (reviews.head(5))
print (reviews.tail(5))
print (reviews.isnull().sum())

reviews = reviews.dropna()
reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time'], 1)
reviews = reviews.reset_index(drop=True)

print (reviews.head(5))
print (reviews.tail(5))

for i in range(5):
    print("Review #",i+1)
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()