# Bipolar Disorder Subtypes and Mood Stabilisers

### An NLP exploration of the two main subtypes of bipolar disorder with regards to Reddit posts on mood stabilisers

--- 

# Executive Summary

Bipolar disorder patients struggle often to manage their disorder and suffer from a variety of side effects from their medication. For those who seek professional medical help, or even in deciding what help to seek, some of them seek help from online communities such as Reddit. Medication is a big part of the online discourse, and to this end, a data science project was undertaken to explore posts from the r/bipolar subreddit mentioning 3 main mood stabilisers for bipolar disorder, namely lithium, lamotrigine (Lamictal) and quetiapine (Seroquel). The project focussed on the differences between the two main subtypes of bipolar disorder.

A Natural Language Processing (NLP) component was involved. This part used exploratory data analysis (EDA) and NLP techniques such as word counts and VADER Sentiment Analysis to derive insights from the data. Negative VADER Sentiment Analysis compound scores indicate negative emotion, and the opposite indictates postive emotion. The exploration revealed that there were more posts with very negative VADER Sentiment compound scores from Bipolar 2 post authors as compared to Bipolar 1 post authors. Compound scores visualised by month showed that there were dips in compound scores for Bipolar 1 authors in April, September and October. There were dips in August and October for Bipolar 2 authors. These dips corresponded to suicidality trends mentioned in scientific literature. The score, when visualised by day of week, showed that Bipolar 2 authors experienced a dip on Tuesdays and Sundays.

When posts mentioning serious topics were tracked by month, there was a dip in April for Bipolar 1 authors, meaning that even though they did not post on serious topics, their sentiment in posts they wrote was detected as more negative than what it usually would have been. Considering the previous findings, this could be taken as a cause for concern. This is because it might indicate that the authors did not think there was anything serious to discuss, but were undergoing mood instability in actual fact. When posts mentioning serious topics were tracked by day of week, there was a spike for Bipolar 2 authors on Sunday and Monday, but a dip on Sunday for Bipolar 1 authors.

Word vectorisation showed that both subtypes are predominantly concerned about side effects of their medication. Also, they seek advice on the subreddit about the first time they had an experience with the medications. Weight gain can also be observed as a side effect that users are highly concerned about. Reddit users tended to seek advice on either alternative mood stabilisers, or the multiple mood stabilisers they were taking. Also, words indicating specific side effects were seen.

The project also involved a classification component to detect the bipolar subtype of authors of posts within the given scope of the data. This was important as post authors have explictly declared their subtype for only a tenth of posts on the subreddit. If a classifier were to be built, this would increase the amount of labelled data available for research. The classifier would be primarily evaluated by its accuracy score, having an accuracy score of at least 0.7 to be considered as a candidate. The CatBoost Classifier on its default settings provided the best results, with a test accuracy score of 0.7. It had a (5-fold) cross-validated accuracy of 0.69 as well as a training accuracy of 0.89. In comparison, the baseline model, Random Forest, had a test accuracy of 0.65. It had a cross-validated accuracy of 0.65 and a training accuracy of 1 (severe overfitting). Other variations of models were tried, such as the CatBoost Classifier stacked with the XGBoost and LGBM Classifer models. The PyCaret library and several Scikit-learn models stacked together (LogisticRegression + KNeighborsClassifier + DecisionTreeClassifier + SVC + MultinomialNB) were also used. These did not perform better than the default CatBoost.

We have merely scraped the surface, and there is still much to be studied from the vast banks of user-generated text available on Reddit. In summary, this project has been a useful exercise to illustrate the potential of data science to reveal insights, as well as providing some practical action points for both clinicians and people with bipolar to better manange this serious condition. The project has revealed the concerns that Reddit users with bipolar post online, as well as laid the foundation for the classification of the 90% of posts on the subreddit which are currently unlabeled with their author's subtype. Future analyses could be done on a different scope, or current analyses could be used to develop material for psychoeducation to help treat bipolar (such as in the form of a Q&A verified by medical professionals).

# Problem Statement

Bipolar disorder is a serious mental disorder that affects 46 million of the world's population, or 0.3-1.2%. It has been observed that many bipolar individuals seek advice from online communitites such as Reddit about their disorder. Also, medication is a big part of the discourse.

This project will involve NLP and data analyses of posts from the r/bipolar subreddit mentioning 3 main mood stabilisers for bipolar disorder, namely lithium, lamotrigine (Lamictal) and quetiapine (Seroquel). As bipolar subtype significantly influences the experience of the disorder and the medications prescribed, the project will focus on the differences between the two main subtypes of bipolar disorder (Type 1 and Type 2).

A classifier will be built to detect the bipolar subtype of authors of posts within the given scope of the data. This is important as post authors have explictly declared their subtype for only a tenth of posts on the subreddit. If a classifier can be built, this could increase the amount of labelled data available for research. The CatBoost Classifier will be used. It will also be stacked with the XGBoost and LGBM Classifer models. The PyCaret library and several Scikit-learn models stacked together will also be used. This classifier will be primarily evaluated by its accuracy score, and it should have an accuracy score of at least 0.7 to be considered as a candidate.

Clinicians and potentially individuals with the disorder will benefit from the information from this analysis.

# Background and Research

Bipolar disorder is a mental disorder that affects 46 million of the world's population, or 0.3-1.2% ([*source*](https://www.nimh.nih.gov/health/topics/bipolar-disorder)). As of 2017, 0.82% of the Singapore population suffers from the condition, and zooming in, 1.07% of the subpopulation of Singaporean 20-24 year olds ([*source*](https://ourworldindata.org/grapher/prevalence-of-bipolar-disorder-by-age?country=~SGP)). In the United States, the corresponding numbers are 0.68% and 0.96% ([*source*](https://ourworldindata.org/grapher/prevalence-of-bipolar-disorder-by-age?country=~USA)).

Suicide is the leading cause of premature death for bipolar sufferers ([*source*](https://www.treatmentadvocacycenter.org/evidence-and-research/learn-more-about/463-bipolar-disorder-fact-sheet)). The suicide rate for bipolar patients is about 10–30 times higher than the corresponding rate in the general population ([*source*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6723289/)). Also, up to 20% of bipolar patients (mostly untreated) end their life by suicide, and 20–60% make an attempt least once in their life ([*source*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6723289/)). Bipolar patients also have an "increased risk of death from all causes" which is double than that expected in the general population ([*source*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4939858/)). People with bipolar died on average between 9 to 20 years younger than people in the general population, and for people with recurrent depression, they died around seven to 11 years younger ([*source*](https://onlinelibrary.wiley.com/doi/full/10.1002/wps.20128)). In comparison, heavy smokers died eight to 10 years younger ([*source*](https://onlinelibrary.wiley.com/doi/full/10.1002/wps.20128)). This means smoking 20 or more cigarettes a day ([*source*](https://www.psychiatryadvisor.com/home/bipolar-disorder-advisor/when-bipolar-disorder-presents-consider-cardiovascular-comorbidities/)).

There are two main subtypes of bipolar disorder. Bipolar I is characterised by both extreme highs and lows (mania and depression), while Bipolar II has predominantly depression episodes, but with less extreme highs (termed as hypomania) ([*source*](https://www.nimh.nih.gov/health/topics/bipolar-disorder)). A survey of 11 countries indicated the lifetime prevalence of Type 1 to be at 0.6% and Type 2 to be at 0.4% ([*source*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3486639/)). Both manic and depressive symptoms can occur simultaneously in the form of a mixed episode, which carries a serious risk of suicide ([*source*](https://www.webmd.com/bipolar-disorder/guide/mixed-bipolar-disorder)). It is important to diagnose the subtype correctly, as the treatment plan will be different depending on which of the symptoms are more dominant ([*source*](https://www.mayoclinic.org/diseases-conditions/bipolar-disorder/expert-answers/bipolar-treatment/faq-20058042)) ([*source*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5310104/)). Mood stabilisers are the mainstay of bipolar treatment, which aim to counter both the depression and the mania (albeit with several side effects) ([*source*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5310104/)) ([*source*](https://www.webmd.com/bipolar-disorder/guide/treating-bipolar-medication)) ([*source*](https://www.nps.org.au/australian-prescriber/articles/mood-stabilisers)) as opposed to a unidirectional effect, such as from antidepressants. The common mood stabilisers are lithium, lamotrigine (brand name: Lamictal), quetiapine (brand name: Seroquel) and valproate ([*source*](https://www.psychiatrictimes.com/view/top-mood-stabilizers-bipolar-disorder)) ([*source*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5310104/)). These drugs have been prescribed for decades ([*source*](https://en.wikipedia.org/wiki/Lithium_(medication))) ([*source*](https://en.wikipedia.org/wiki/Lamotrigine)) ([*source*](https://en.wikipedia.org/wiki/Quetiapine)) ([*source*](https://en.wikipedia.org/wiki/Valproate)). Lithium is an antimanic drug, lamotrigine and valproate are anticonvulsants and quetiapine is an antispsychotic ([*source*](https://www.webmd.com/bipolar-disorder/guide/treating-bipolar-medication)). These drugs are now being used as mood stabilisers ([*source*](https://www.webmd.com/bipolar-disorder/guide/treating-bipolar-medication)).

Bipolar disorder is possibly left untreated in half of diagnosed individuals in any given year ([*source*](https://www.treatmentadvocacycenter.org/evidence-and-research/learn-more-about/463-bipolar-disorder-fact-sheet)). However, the associated stigma and discrimination may prevent sufferers from seeking help ([*source*](https://www.who.int/mental_health/management/info_sheet.pdf)). The disorder is "associated with significant impairment in work, family and social life, beyond the acute phases of the illness" ([*source*](https://www.karger.com/Article/Abstract/228249)). Bipolar disorder is frequently misdiagnosed as depressive disorder, which may result in an exacerbation of their condition due to the prescription of antidepressants ([*source*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2945875/)). This is because patients tend to seek medical attention only when they are in the depressive phase of the illness ([*source*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2945875/)). The aforementioned mood stabilisers are prescribed as a treatment for bipolar depression, with antidepressants only as a last resort ([*source*](https://onlinelibrary.wiley.com/doi/full/10.1111/bdi.12860)). Treatment satisfaction is important for one's perspective about being able to cope with bipolar disorder ([*source*](https://www.dbsalliance.org/education/bipolar-disorder/bipolar-disorder-statistics/)). People with bipolar can face up to ten years of struggling with the disorder before they can be diagnosed as having bipolar ([*source*](https://www.dbsalliance.org/education/bipolar-disorder/bipolar-disorder-statistics/)). Therefore, it is important to have conversations about bipolar medications, and about diagnosis of bipolar types as well.

The data for this project will come from Reddit. About half of Reddit users come from the United States, and about two-thirds of them are male ([*source*](https://www.alphr.com/demographics-reddit/)).  Almost two-thirds of active US-based users are in their 20s and 30s ([*source*](https://www.statista.com/statistics/1125159/reddit-us-app-users-age/)). 

# Data sources

The data was retrieved from the r/bipolar subreddit using the Pushshift API.

# Data Dictionary

The data dictionary is as follows:

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**text**|object|sb|Combined text of the title and post body.|
|**text_num_words**|int64|sb|Number of words in the combined text of the post.|
|**year_**|int64|sb|Year the post was published.|
|**month_**|int64|sb|Month the post was published.|
|**day_**|int64|sb|Day of the month the post was published.|
|**week_**|int64|sb|Week of the year the post was published.|
|**day_of_week_**|int64|sb|Day of the week the post was published.|
|**hour_**|int64|sb|Hour of the day the post was published.|
|**neg**|int64|sb|The negative score from VADER Sentiment Analysis.|
|**neu**|int64|sb|The neutral score from VADER Sentiment Analysis.|
|**pos**|int64|sb|The positive score from VADER Sentiment Analysis.|
|**compound_**|int64|sb|The compound score from VADER Sentiment Analysis.|
|**serious_**|int64|sb|Indicator for if the topic of the post was serious or crisis-related.|
|**bp_type**|int64|sb|Bipolar type of the post author.|
|**lithium_**|int64|sb|Indicator for if the post contains the word 'lithium'.|
|**lamotrigine_**|int64|sb|Indicator for if the post contains the words 'lamotrigine' or 'lamictal.|
|**quetiapine_**|int64|sb|Indicator for if the post contains the words 'quetiapine' or 'seroquel'.|

# Results and Analysis

The models with their details and metrics are as follows:

|Model|Vectorization|Test Accuracy|Cross-Val Training Accuracy (5-fold)|Training Accuracy|F1 Score (Bipolar 1)|F1 Score (Bipolar 2)|ROC AUC|
|---|---|---|---|---|---|---|---|
|**CatBoost Classifier**|Count|0.72|0.69|0.89|0.61|0.78|0.78|
|**Random Forest**|Count|0.65|0.65|1.00|0.44|0.75|0.71|
|**CatBoost + XGBoost + LGBM Classifiers**|TFIDF|0.71|0.67|0.68|0.59|0.78|0.68|
|**LogisticRegression + KNeighborsClassifier + DecisionTreeClassifier + SVC + MultinomialNB**|TFIDF|0.69|0.65|0.99|0.03|0.73|0.71|

The results indicate that the CatBoost Classifier is on par or superior in metrics other than accuracy. For example, it has better F1 scores as well. The cross-validated score is also very close to the test score. Combined with the fact that it has an accuracy above 0.7, it is thus the model of choice. 

# Conclusion and Recommendations

In summary, we have achieved a model that has a test accuracy of 0.7. The chosen model was CatBoost Classifier on the default parameters. There were efforts to optimise the model, but it turned out that the winning model was the un-optimised one. Further analyses showed that the problem was not easy to solve. To date, no similar classifier has been built. Existing classifiers focussed more on diagnosing the presence and absence of the disease, rather than the subtype.

From the EDA, we have gained some insights that would be useful to both clinicians and individuals with bipolar. Notably, there are some time-based patterns that were observed. For Bipolar Type 1, these individuals should be monitored during April especially due to the risk of manic or mixed instability. They should also be monitored during September to October. For Bipolar Type 2, these individuals should be monitored during Auguest and October. They should also be monitored on weekends. The main risk for Type 2 individuals would be depressive episodes. For both types, individuals had concerns with side effects, in particular, weight gain. It appeared that users of the subreddit wanted to compare their experience with that of others and find out more information.

To develop the project further, these analyses could be done using a different scope. For example, problems with family or employment could be investigated by changing the search keywords when pulling the data. Next, the main concerns surfaced on the subreddit could be used to form Q&A material that could be provided by clinicians to patients, so that advice on the condition can be verified by medical professionals. This is part of a strategy to counter the disorder, called psychoeducation. Also, if the concerns raised on the subreddit are quite different from what they have discussed with patients, clinicians might be alterted to concerns that patients are not comfortable to tell them directly. Clinicians would also be better attuned to specific concerns about the medications studied. Lastly, if the classifier reaches a high enough accuracy, it could be used to label the approximately 90% of posts where post authors have not explicitly declared their bipolar subtype. This would increase the resolution of insights that we can glean from this treasure trove of textual data.

We have merely scraped the surface, and there is still much to be studied from the vast banks of user-generated text available on Reddit. In summary, this project has been a useful exercise to illustrate the potential of data science to reveal insights, as well as providing some practical action points for both clinicians and people with bipolar to better manange this serious condition.