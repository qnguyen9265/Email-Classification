# Email Classification

I created a Python script named classi.py to classify my emails. I decided to classify emails based on shopping/advertisements and otherwise. All of my on-topic emails are marketing a product and attempting to entice me to spend money, while all of my off-topic emails do not. Off-topic emails may include news, updates, letters from the government, or anything else. This is a reupload of a project done in 2022 for a Python coding course with all references to course material removed.

## Explanation and Reasoning

Below, I explain my thoughts and reasoning while working through this project.

My email datasets are in: 
- testing/
	- off-topic
	- on-topic
- training/
	- off-topic
	- on-topic

While copying the text from my emails, I ended up also copying and pasting the pictures as well, which turned into the text describing the image if there was some sort of description available. I also deleted any mentions relating to myself, people I know, my personal email account, or any other personal information that I could find. A few of my official loan provider emails had part of my account number shown which had to be deleted. No HTML tags were included in any of my copied emails.

|Testing Email|Naive Bayes Classification|Actual Classification|
|-------------|--------------------------|---------------------|
|off-topic-1.txt|off-topic|off-topic
|off-topic-2.txt|off-topic|off-topic
|off-topic-3.txt|off-topic|off-topic
|off-topic-4.txt|off-topic|off-topic
|off-topic-5.txt|off-topic|off-topic
| on-topic-1.txt|marketing|marketing
| on-topic-2.txt|marketing|marketing
| on-topic-3.txt|marketing|marketing
| on-topic-4.txt|off-topic|marketing
| on-topic-5.txt|marketing|marketing

The classifier only incorrectly classified one email: on-topic-4.txt. The particular email is about a limitied release of pens. The classifier may have incorrectly classified the email because while it uses phrases like "LIMITED RELEASE" and "Shop Now", the email doesn't make any mention of sales, discounts, or Black Friday, which many of my training emails do.

While reading in my emails for training, I had an error due to some characters in my emails not being readable by Python by default. I fixed this by specifying the encoding of the email and removing any emojis.

```Python
file = open(filename, 'r', encoding='utf8')
```

### Confusion Matrix:

|   |   |Actual||
|-------------|---------|---------|---------|
|**Predicted**|         |marketing|off-topic|
|             |marketing|4 (TP)|0 (FP)|
|             |off-topic|1 (FN)|5 (TN)|

The classifier performed very well using my training and testing sets. Only one email was incorrectly classified, which was on-topic-4.txt getting marked as a false negative. A 90% is still an A.

Generally, I would prefer that a classifier have more false positives than false negatives. It is better to have to manually sort through false positives for true positives than to have false negatives that may cause issues.
In the case of emails, if an email is incorrectly marked as spam and you check your spam folder, you can review those false positives and move emails that were incorrectly marked spam back into the correct location.
On the other hand, this does run the risk of losing non-spam emails, as many email providers periodically delete spam emails. I have personally checked my spam folders on occaision only to see a few legitimate emails and nothing else.
No system is perfect, so having a way to review results is beneficial.

## References

https://www.sciencedirect.com/topics/engineering/confusion-matrix

https://stackoverflow.com/a/9233174
