# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?
The accuracy drops from 0.88 to 0.625 this is because without the scaler the model is likely skewed by income the most, since it has the highest value.
2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.
It is only 62.5% accurate. This is not accurate enough because the model essentially has as good of a chance of getting it right as guessing randomly.
3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?
There wasn't really a pattern because the model was incorrect for such a small portion of the data points that there wasn't enough wrong data to make conclusions.
4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.
She would purchase the SUV according to the model.
