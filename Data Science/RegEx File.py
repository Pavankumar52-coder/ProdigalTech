# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OOvp0wq4RfcIG2sc3sQw5bpyekQYhDgy

EXERCISES
### **Question 1**

**Part** **A:**

**True Positives (TP):**

- i wouldn't want this to end up on your credit report
- this will affect your credit report
- yes it is going to affect your credit report

**False Positives (FP):**

- no this will never affect your credit report
- this will not end up on your credit report
- yes it is not going to affect credit report
"""

(i would(n't| not)|yes it is|this will) (\S+\s){0,3}(affect|end up on) your credit report

"""2. Write a regex to capture all the False Positive phrases:"""

(no this will|not end up|never|not going to affect|yes it is not) (\S+\s){0,3}credit report

"""3.Write a single regex that captures most TPs and eliminates most FPs:"""

(i would(n't| not)|yes it is|this will) (\S+\s){0,3}(affect|end up on) your credit report

"""4.Report the Precision and Recall"""

import re

# Sample set
true_positives = [
    "i wouldn't want this to end up on your credit report",
    "this will affect your credit report",
    "yes it is going to affect your credit report",
]
false_positives = [
    "no this will never affect your credit report",
    "this will not end up on your credit report",
    "yes it is not going to affect credit report",
]
sample_set = true_positives + false_positives

regex = r"(i would(n't| not)|yes it is|this will) (\S+\s){0,3}(affect|end up on) your credit report"

# Evaluate
tp = sum(1 for phrase in true_positives if re.search(regex, phrase))
fp = sum(1 for phrase in false_positives if re.search(regex, phrase))
fn = len(true_positives) - tp

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f"Precision: {precision}")
print(f"Recall: {recall}")

"""PART-B
1. Phrases captured by Regex 1:
Examples:
a. Verify your social security number.
b. Provide your date of birth.
c. Verify your address.
d. Provide your social security details.
e. Verify your date of birth.

2. Phrases captured by Regex 2:
Examples:
a.Provide your credit card number.
b.Provide your bank account details.
c.Provide debit card details.
d.Credit card account number, please.
e.Provide your account number.

QUESTION-2
**Write regexes to retain as many True Positives for each of the pairs as possible and eliminate their corresponding False Positives as below:**

**Part A**

**True Positives (TP)**

- yes it is going to affect your credit report
- would hate for it to just you be put on your credit report
- before we place all your credit report yes
- i wouldn't want this to end up on your credit report
- understand will keep off credit report and everything

**False Positives (FP)**

- yes it is not going to affect credit report
- so you won't get credit reported or anything okay
- important matter is that it plays nothing on your credit reports
- this matter has that no effect on your credit report
- no please note that it has not been placed on your credit reports

Write a regex to retain as many TPs as possible while eliminating as many FPs as possible for Part A.

---
"""

(yes it is|i wouldn't want|would hate for it|before we place|keep off) (\S+\s){0,3}credit report

"""**Part B**

**True Positives (TP)**

- before this unpaid dues affects your credit report
- then after that it goes you know to your credit report
- someone want to fetch your credit report
- small amount like that to go on your credit report

**False Positives (FP)**

- no this will never affect your credit report
- just remember no one will take it to your credit report
- nothing goes to your credit report
- we don’t report to credit bureau

Write a regex to retain as many TPs as possible while eliminating as many FPs as possible for Part B.
"""

(yes it is|i wouldn't want|would hate for it|before we place|keep off) (\S+\s){0,3}credit report

"""Question-3
Deploy a LLM based tool which accepts a simple input and generates a meaningful output.
"""