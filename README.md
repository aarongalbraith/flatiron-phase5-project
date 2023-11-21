# Predicting Birth Control Sentiment
#### Aaron Galbraith
#### Flatiron Data Science Capstone Project
https://www.linkedin.com/in/aarongalbraith \
https://github.com/aarongalbraith
#### Submitted: November 21, 2023

![alt text](images/title.jpeg)
![alt text](images/title.jpeg)

## Overview

Pfizer seeks better information about the sentiment of its potential customer base toward various prescription birth control methods. We analyzed user-generated reviews of birth control drugs from drugs.com and made recommendations to Pfizer based on our findings. Pfizer can track developing trends by using our modeling tool to analyze conversations happening elsewhere online.

## Business and Data Understanding

### Business Understanding

Following the US Supreme Court ruling in [Dobbs (2022)](https://www.supremecourt.gov/opinions/21pdf/19-1392_6j37.pdf), many states began changing laws regarding reproductive health care rights. In the new reproductive environment created by this ruling, Americans who are concerned with family planning are showing greater interest in birth control options and are more likely to consume and practice the birth control methods (that remain legal) in greater numbers than before. Pfizer can capitalize on this trend by understanding public perceptions of the various methods and responding to these perceptions in their marketing.

In 2018, researchers Surya Kallumadi and Felix Gräßer at UC Irvine created the [UCI ML Drug Review Dataset](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018/) after collecting reviews from [Drugs.com](https://www.drugs.com/) that users had written about various drugs between 2008 and 2017. A substantial portion of these reviews addressed birth control and emergency contraception drugs.

People often share similar sentiments with each other in online spaces such as [Reddit](https://www.reddit.com/r/birthcontrol/) and [Quora](https://www.quora.com/search?q=birth%20control). Our project analyzes the Drug Review Dataset in order to 1) learn what the Dataset can tell Pfizer about sentiments toward the various methods of birth control and 2) train a model that can be applied in other online spaces to determine what birth control methods users are discussing and how they feel about them. With this tool, Pfizer can more effectively market their products to the increased demand created by the Dobbs ruling.

### Data Understanding
After a substantial amount of cleaning, the data set included 21,779 records. Each record had these features:
#### Drug Name
These labels varied greatly. Many were specific brand names, while others were generic or chemical names, or even combinations of chemical names. 
#### Condition
This feature had many missing labels. We eventually trimmed this feature to just two labels: "Birth Control" and "Emergency Contraception". In fact there was some cross-mixing of these two conditions, i.e. records labeled "Birth Control" that actually reviewed drugs for emergency contraception purposes and vice versa.
#### Method (of Birth Control)
This feature was central to our analysis, but it was not provided in the original dataset. We had to construct it by matching the various brand drug name labels to their corresponding method of birth control. ChatGPT was instrumental in the process of gathering this data.

We do not consider emergency contraception pills to be in direct competition with the other methods of birth control, so this method label was sometimes included in analysis and visual displays when comparisons seemed useful or when confusion was possible and otherwise excluded.
#### Review
This feature was the text of the review that a user posted on drugs.com. There were a great deal of duplicate reviews, as explained further below.
#### Rating
![rating distribution](images/plot_rating_dist.png)
Users submitted a rating between 1 and 10 accompanying each review.
#### Date
![date distribution](images/plot_dates.png)
The records spanned from February 2008 to November 2017. The number of reviews surged in 2014.
![rating comparison over time](images/plot_rating_time.png)
The average ratings of the methods vary over time.
#### Useful Count
This feature counted the number of "upvotes" recorded by other users. This did not factor into our analysis. In further inquiry, it would be wise to note that the increase in the number of records from 2014 onward likely correlates with an increase in upvotes that does not necessarily reflect *better* reviews but simply *more* of them. Any analysis of this feature should perhaps calculate upvotes as a percentage of the total upvotes during a certain timespan, such as a day or a month.
## Data Preparation
### Duplicates, drug names, and missing condition labels
The majority of the records were entered twice: once with a brand name in the `drugName` feature and once with a generic or chemical name. *Some* of these duplicates had *one* missing condition label. By recognizing the nature of these special pairs, we were able to restore many of the missing condition labels (by matching them with their pair-mate).

For the remaining missing condition labels, we assigned the label that most commonly corresponded with the drug name listed. For example, if a record specified a drug name of "Viagra" but had no condition label, we would assign it the condition of "Erectile Dysfunction", as that was the most common condition associated with Viagra.

Once we had successfully restored as many missing condition labels as possible, we dropped the remaining records with missing condition labels and further dropped all records with condition labels other than "Birth Control" or "Emergency Contraceptive".

There were still more duplication instances beyond the special brand/generic pairs described earlier. This involved instances of the same review (unmistakably verbatim) appearing in multiple records, sometimes on different dates, usually with differing numbers of upvotes. We assumed in these cases that the same user had posted a review multiple times. We collapsed these reviews into a single record and modified the `usefulCount` to reflect the *total* number of upvotes from all instances. In at least one case, a single representative `date` label had to be chosen arbitrarily from two options that were only one day apart.

## Exploration

![use percentage plot](images/plot_use_pct.png)
This shows that the pill is the dominant method, and IUDs are the next most commonly used, while most other prescription methods are not very commonly used.
![relative sentiment distribution](images/plot_sent_dist_2.png)
This shows that some of the lesser used methods (vaginal rings and patches) enjoy quite favorable ratings. The pill is not especially highly rated. Injectables and implantables are rated the lowest of all methods.

## Modeling

### Dummy Classifier (BASELINE)

### Decision Tree

### Logistic Regression

### Summary of Model Performance

We experimented with some other models as well, but none of the results were as relevant as the main three mentioned above.

| Model | Training Accuracy | Test Accuracy |
| -------- | ------- | ------- |
| Logistic Regression (rough) | 95.9% | 62.6% |
| Logistic Regression (oversampled) | 95.8% | 61.7% |
| Naive Bayes (rough) | 79.4% | 71.5% |
| Naive Bayes (tuned) | 89.0% | 72.2% |
| Naive Bayes (oversampled) | 86.7% | 68.0% |
| Decision Trees (rough) | 96.5% | 70.5% |
| Decision Trees (tuned) | 79.3% | 71.6% |
| Bagged Trees | 74.7% | 71.7% |
| Random Forest (rough) | 96.5% | 73.2% |
| Random Forest (tuned) | 86.4% | 72.5% |
| Support Vector Machine | 92.6% | 74.3% |
| Adaboost | 74.1% | 70.6% |
| Gradient Boost | 74.9% | 72.3% |
| XG Boost | 84.1% | 73.3% |

### Confusion Matrix for Final Model

## Evaluation

## Recommendations
#### 1. Consider developing and marketing a patch as an alternative to the pill
Users rate the patch quite highly, but it is used much less than the pill. There is room for growth here.
#### 2. Emphasize favorable weight and body image effects of patch methods
Our word clouds and term importance analysis shows that users appreciate these aspects of the patch.
#### 3. Abandon injectable product (Depo-Provera)
This method is not commonly used and not highly rated.
#### 4. Apply our prediction models in online spaces
Look in spaces such as Reddit and Quora to gather updated data on public sentiment toward the various birth control methods.

## Further Inquiry

- Incorporate upvotes more into analysis and modeling
- Explore the possibility of a recommendation algorithm
- More feature engineering possibilities
    - whether words are in English (spelled correctly)
    - use of emoticons
    - whether it uses slang sex terms versus technical language
- Analyze evidence of reviewers switching methods or brands
- Understand what went wrong with XGB reproducibility in the multiclass model
- Scrape new text from Reddit, Quora and apply modeling to it

## Links to PDFs

Find the notebook [here](https://github.com/aarongalbraith/flatiron-phase5-project/tree/main/deliverables/notebook.pdf)

Find the presentation [here](https://github.com/aarongalbraith/flatiron-phase5-project/tree/main/deliverables/presentation.pdf)

Find the github repository [here](https://github.com/aarongalbraith/flatiron-phase5-project/tree/main/deliverables/github.pdf)

Find reproducibility notes and instructions to run the notebook [here](https://github.com/aarongalbraith/flatiron-phase5-project/tree/main/Reproducibility%20Notes.md)