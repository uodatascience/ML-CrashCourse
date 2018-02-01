---
title: "Regularized Regression in Caret"
author: "Cory Costello"
date: "January 24, 2018"
output: 
  md_document:
    preserve_yaml: true
    toc: true
    toc_depth: 2
---

# What is Regularized regression?

This is intended to provide a (very brief) introduction to some key concepts in Machine Learning/Predictive Modeling, and how they work within a regression context. Regularized regression is a nice place for folks with a psych background to start, because its an extension of the familiar regression models we've all come to know and love.

Regression models, of course, are used when you have a *continuous* outcome. So many of our problems in psych involve continuous outcomes (e.g., personality dimensions, emotions/affect, etc.), and so regression models are pretty useful for psychologists. 

But, before we dive into regularized regression, we should first cover some basic concepts in machine learning.

## The Bias-Variance Tradeoff, Overfitting, and Cross-Validation

### Bias-Variance Tradeoff

This came up briefly last week, but the bias-variance tradeoff is really crucial to everything we'll talk about today. The basic idea is that you can partition error into two components: bias and variance. **Bias** refers to the extent to which a model produces parameter estimates that miss in a particular direction (e.g., consistently over-estimating or consistently under-estimating a parameter value). **Variance** refers to the extent to which a model produces parameter estimates that vary from their central tendency across different datasets. 

There is a tradeoff between bias and variance: all else equal, increasing bias will decrease variance. The basic idea here is that if we have a zero-bias estimator, it will tend to try to fit everything in the data, whereas an estimator biased in some direction won't. This isn't to say that bias is good or bad, just that there are times where we might want to increase bias to decrease variance. As we'll see later, regularization is basically one method for introducing bias (to minimize variance).

### Overfitting 

This is probably familiar to folks in here (and it came up last week), so I won't say much about it here. The basic idea is that any data has signal and noise. Sometimes, something appears to be signal but is actually noise. That is, when our statistical models are searching for the best solution, they sometimes will be fooled into thinking some noise is signal. This is usually called **overfitting**, and it has presented a pretty substantial problem in statistical modeling. As you'll see, one of our goals is to try to avoid overfitting. Also worth noting is that overfitting will tend to produce a model with high variance, because noise will vary from dataset to dataset (basically by definition), and so a model which has fit noise will not do well across different datasets (with different noise).

### Cross-Validation

Cross-validation generally refers to taking a model that you trained on some data and using it in a new dataset. Unlike a replication, the model parameters carry over from the training to the test data (i.e., you don't simply use the same variables and re-estimate the model parameters; you save the model parameters, and use it to predict the outcome variable). You can use cross-validation both to train and evaluate a model. A simple example may make this clear.

Let's say we think home size (in square-feet) is the only relevant predictor for house price. So, we have some data on prices of recently sold houses, and estimate a model predicting house price from square-feet:

$$\hat{price_{i}} = b_0 + b_1*sqaurefeet_i$$

Let's say we get these parameter values:

$$\hat{price_{i}} = 100 + 50*sqaurefeet_i$$

And now we want to cross-validate in a hold-out sample. We wouldn't simply estimate this model again:

$$\hat{price_{i}} = b_0 + b_1*sqaurefeet_i$$

We would instead apply this model:

$$\hat{price_{i}} = 100 + 50*sqaurefeet$$
And evaluate how well it did. We could do this either by how much it misses, which is usually done with root mean squared error (RMSE). This is the average squared difference between observed ($y_i$) and expected ($\hat{y}_i$) values:

$$MSE = \frac{1}{n}\sum\limits_{i=1}^{n}(y_i - \hat{y_{i}})^2$$

And in our example, the y variable is house price:

$$MSE = \frac{1}{n}\sum\limits_{i=1}^{n}(price_i - \hat{price_{i}})^2$$
And then finally, we take the square root for RMSE:

$$RMSE = \sqrt{MSE}$$

Typically, people will also look at prediction accuracy, using the model's $R^2$. This is interpreted the same way as $R^2$ always is (as the % of variance in the outcome accounted for by the model).


#### K-fold cross-validation

There are different varieties of cross-validation. The most intuitive version is to create a single partition of data (i.e., split full data frame into two dataframes: training and test). However, there are other methods for cross-validation. One that has been gaining steam (or is maybe already at full steam at this point) is **k-fold cross-validation**. The basic idea is that we split a dataset into k subsamples (called folds). We then treat one subsample as the holdout sample, train on the remaining subsamples, and cross-validate on the holdout sample; then rinse and repeat so to speak. An example will probably help here.

Let's take a ridiculously simple example (based on the earlier example). We want to predict house sale price from square footage:

$$\hat{price_i} = b_0 + b_1*sqaurefeet_i$$

Let's say we have just 30 cases, and we use 10-fold cross validation. Let each observation be indicated by $o_i$, so the first observation is $o_1$, the second is $o_2$, and the third is $o_3$, etc. 

First, we would fit a model using folds 2 through 10 (i.e., $o_4$ to $o_30$), and then test it on the first fold ($o_1$ to $o_3$). Then, we would fit the model using folds 1 and 3-10 (i.e., $o_1$ to $o_3$ & $o_7$ to $o_30$) and test it on the 2nd fold ($o_4$, $o_5$, $o_6$), and so on until each fold was used as the holdout sample.

Then, we calculate the average performance across all of the tests.

Note that you can also use k-fold cross-validation for training purposes. Basically, this works by taking the best fitting model from a k-fold cross-validation procedure, and then testing it on a new holdout sample. 


## Regularization

Now let's get to regularized regression. This is a pretty simple extension of OLS regression. The logic of it is basically that OLS regression is minimally biased, but because of this, is higher variance than we might want. So, the solution is to introduce some bias into the model that will decrease variance. This takes the form of a new *penalization*, which tends to either be focused on parameter size, number of parameters, or both. Let's start with the first. I find it helpful to think of these as having different beliefs, and choosing one depends on whether or not those beliefs seem correct.

## Ridge: all of these features matter, but only a little bit.

Ridge regression is basically OLS regression with an extra term. As a refresher, OLS seeks to minimize the sum of squared error, or:

$$SSE = \sum\limits_{i=1}^n (y_i - \hat{y_i})^2$$

Ridge adds an additional penalty:

$$SSE_{L2} = \sum\limits_{i=1}^n (y_i - \hat{y_i}^2) + \lambda \sum\limits_{j=1}^p \beta^2_j$$
This makes it so that paramter values are only allowed to be large if they reduce error enough to justify their size. Functionally, this makes it so parameter values shrink towards 0. You can hopefully see this in that as our paramater values (our betas) increase in size, error increases, since we are adding the sum of squared beta values, times some constant $\lambda$, to our error term SSE. So, unless the parameter values decrease the first part of the error term (the ordinary sum of squared error; to the left of our new penalty) proportionally to their magnitude, they are shrunk toward 0. 

The extent to which they are shrunk towards 0 depends on the value of $\lambda$; higher values lead to more shrinkage than lower values. This is called a *hyperparameter* because it's a parameter that governs other parameters. You can think of $\lambda$ as sort of the cost associated with larger parameter values: higher values of lambda are like telling your model that larger parameter values are more costly (so don't make them large for nothing).

You can think of this penalty as introducing a specific type of bias: bias towards smaller parameter values. However, since larger parameter values can result from overfitting, this bias can result in reducing variance.

So why does Ridge do that and why is it useful? As I said earlier, I find it useful to think of statistical tools as having certain beliefs, and as being useful when those beliefs seem more or less true (in some particular case). Ridge believes that all of the variables you're considering matter, but that most of them matter very little. Put differently, it believes that each variable you've entered belongs in the model, but that most or all only have small contributions. Because of this, people often say that ridge doesn't perform *feature selection*, and shouldn't be used if you need to select features (i.e., variables). This makes sense once you think of what Ridge believes: it believes every variable you're telling it to use should be in the model, but many will simply have small impacts. If we want to select features (i.e., decide what variables go in our model), we need a different tool with a different set of beliefs.

## Lasso: only some features matter, and they might matter a lot

Another popular form of regularized regression is the *least absolute shrinkage and selection operator* model, or *lasso*. Unlike ridge, lasso's regularization simultaneously performs feature selection and model improvement. 

Just like ridge, lasso is essentially our old friend OLS regression with an extra term added to error, which penalizes non-zero parameter values: 

$$SSE_{L1} = \sum\limits_{i=1}^n = (y_i - \hat{y_i}^2) + \lambda \sum\limits_{j=1}^p |\beta_j|$$

It's sort of hard (at least for me) to have a strong intuition about why this simple change leads to a model that functions differently. But, the basic idea is that penalizing the absolute value leads to some parameters actually being set to zero; the idea (I think) is that penalizing the absolute value leads to small departures from zero (e.g., .1) to be relatively more penalized than when you're penalizing the squared value (since squaring a value < 1 leads to a smaller value than its absolute value). This is most consequential for correlated predictors: Ridge will allow each of k correlated predictors to basically share the predictive duty, whereas Lasso will tend to pick the best and ignore the rest. SO, just like with ridge, lasso introduces bias, and its bias is that many predictors will have no relation to the outcome variable (i.e., only some features matter).

Let's walk through an example with correlated predictors that I think will help. Let's say we have an outcome $Y$, and two predictors $X_1$ and $X_2$. And let's imagine $X_1$ and $X_2$ are highly correlated ($r_{X_1, X_2} = .90$). Let's say a model (Model 1) that contains predictors gives us this solution:

$$Model 1: y_i = .40*X_1 + .40*X_2$$
According to the path algebra, including just one of these predictors, $X_1$, in the model would give us the following:

$$Model 2: y_i = .76*X_1$$

Note, this is just the path from $X_2$ to $Y$ (.40) times the correlation between $X_1$ and $X_2$ (.90). So how would each of these penalties treat this? Let's walk through it:

First, let's simulate some data that has the properties we just mentioned. We'll do this with the `mvtnorm` library. This allows us to take a random sample from a multivariate normal distribution. It requires a sample size, a vector of means (equal to the number of variables), and a variance-covariance matrix (called sigma; where r = c = number of variables). Since we're talking about standardized solutions, we'll create a variance-covariance (sigma) matrix that is standardized (i.e., a correlation matrix), with 1's along the diaganol. We then specify the bivariate correlation between each variable, which will be .9 for X1 and X2, and then .76 for X1's relation with Y and .76 for X2's relation with Y (just like above). That looks something like this:

```r
# Load the mvtnorm library
library(mvtnorm)

# specify sigma matrix;
# again, this is the correlation matrix for the variables
# since we're working with standardized values.
sigma <-matrix(c(1, .9, .76,
                 .9, 1, .76,
                 .76, .76, 1), ncol = 3)

# Now take the sample, call it sample_data
sample_data <- data.frame(rmvnorm(n = 1000000, mean = c(0, 0, 0), sigma = sigma))

# give the columns names; we'll use x1, x2, and y just like the example above
names(sample_data) <- c("x1", "x2", "y")
```

Alright, let's check the correlation matrix to make sure we did this correctly. This should match sigma:

```r
cor(sample_data)
```

```
##           x1        x2         y
## x1 1.0000000 0.9000185 0.7597924
## x2 0.9000185 1.0000000 0.7596805
## y  0.7597924 0.7596805 1.0000000
```

Ah, it does! Good, let's proceed. Now if we estimate, a regression with both variables, we should get two beta weights of about .40:


```r
model_1 <-lm(y ~ x1 + x2, data = sample_data)
summary(model_1)
```

```
## 
## Call:
## lm(formula = y ~ x1 + x2, data = sample_data)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -3.00236 -0.42277  0.00042  0.42262  2.85982 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -0.0005922  0.0006261  -0.946    0.344    
## x1           0.4002507  0.0014367 278.593   <2e-16 ***
## x2           0.3992187  0.0014370 277.815   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.6261 on 999997 degrees of freedom
## Multiple R-squared:  0.6076,	Adjusted R-squared:  0.6076 
## F-statistic: 7.741e+05 on 2 and 999997 DF,  p-value: < 2.2e-16
```

Okay, that worked as expected; we get two beta weights of about .40 (if you round to 2 decimals). Now let's check model 2, where we just include 1 x variable (x1). We should get a single beta weight of about .76.


```r
model_2 <-lm(y ~ x1, data = sample_data)
summary(model_2)
```

```
## 
## Call:
## lm(formula = y ~ x1, data = sample_data)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -3.3881 -0.4384  0.0003  0.4390  2.9920 
## 
## Coefficients:
##               Estimate Std. Error  t value Pr(>|t|)    
## (Intercept) -0.0003750  0.0006498   -0.577    0.564    
## x1           0.7594768  0.0006499 1168.612   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.6498 on 999998 degrees of freedom
## Multiple R-squared:  0.5773,	Adjusted R-squared:  0.5773 
## F-statistic: 1.366e+06 on 1 and 999998 DF,  p-value: < 2.2e-16
```

And we do. Now let's walk through the two penalties we covered so far, Ridge and Lasso. Based on what we know so far, Ridge should prefer Model 1 (with x1 and x2) and LASSO should prefer model 2 (the one with just x1)



```r
# pull out the coefficients for each parameter of the two models
model_1_b1 <- model_1$coefficients["x1"]
model_1_b2 <- model_1$coefficients["x2"]
model_2_b1 <- model_2$coefficients["x1"]
# Note, we're not actually estimating b2 in model 2, but this is equivalen to saying its 0
model_2_b2 <- 0

# set lambda to a constant; we'll use .1
lambda <- .1
ridge_penalty_Model_1 <- lambda*(sum(c(model_1_b1^2, model_1_b2^2)))
ridge_penalty_Model_2 <- lambda*(sum(c(model_2_b1^2, model_2_b2^2)))

ridge_penalties <- rbind(ridge_penalty_Model_1, ridge_penalty_Model_2)

lasso_penalty_Model_1 <- lambda*(sum(c(abs(model_1_b1), abs(model_1_b2))))
lasso_penalty_Model_2 <- lambda*(sum(c(abs(model_2_b1), abs(model_2_b2))))

lasso_penalties <- rbind(lasso_penalty_Model_1, lasso_penalty_Model_2)
penalties <- cbind(ridge_penalties, lasso_penalties)

colnames(penalties)<- c("ridge", "lasso")
row.names(penalties) <- c("model 1", "model 2")
knitr::kable(penalties, digits = 3)
```

           ridge   lasso
--------  ------  ------
model 1    0.032   0.080
model 2    0.058   0.076

Now what you could hopefully see there is that, all else equal, lasso prefers fewer predictors (which can have larger values) than ridge. How much it penalizes predictors depends again on $\lambda$, which again is a *hyperparameter*.

So returning to why we would use it, it's easiest for me to see when it would be useful by thinking about what Lasso believes: it believes that non-zero predictors are costly (and cost doesn't accelerate with parameter value size, like ridge does). It (sort of) believes that only some of the variables are needed, and the ones that are needed can take on relatively larger sizes. 

What if our belief is somewhere in between these options: that some variables may not be needed (may actually be zero), but that many of the variables should have smaller values?

## Elastic Net: maybe everything matters, and maybe only a little bit.

Elastic net combines the penalties used by ridge and lasso. In doing so, it basically takes the middle ground between these two methods: penalizing non-zero values (feature selection) and penalizing values the further they depart from zero (regularization). So now, our error has three terms: 

1) sum of squared errors
2) ridge penalty
3) lasso penalty

The formula for this error term is:

$$SSE_{Enet} = \sum\limits_{i=1}^n  (y_i - \hat{y_i}^2) + \lambda_1 \sum\limits_{j=1}^p \beta^2_j + \lambda_2 \sum\limits_{j=1}^p |\beta_j|$$

Basically, elastic net is sort of a best of both worlds approach: it gives you the feature selection of lasso, and regularizes as effective as ridge. It thus introduces two dimensions of bias: 

1) that most predictors have small relations to the outcome. 
2) that many predicotrs have no relation to the outcome.

How much each is priortized depends on the sie of $\lambda_1$ and $\lambda_2$ respectively. It's often a great place to start, because as you're tuning the hyperparameters, you can get to one of the other methods if that is truly the best method. For example, if lasso is actually the best method for your data, then (if your training is working well) you should end up with a zero value for $\lambda_1$, leading to the ridge penalty dropping out of the model (and leaving you with a lasso model). However, in my limited experience, it usually ends up with some non-zero value for both (which I think says something about the problems we deal with).

In terms of beliefs, Elastic net is basically a more flexible thinker: it thinks we might only need few predictors and that each predictor may only contribute a little bit, and its willing to weigh these things more or less depending on what works better (either determined a priori, or determined via training).

Okay, this has been a (very brief) intro to regularized regression and some foundational concepts in machine learning necessary to understand it.

## Example using Caret

Now let's walk through an example:

We're going to work with this data on wine reviews. It has the score it received in a rating, as well as some data about the wine, and a description of the wine. We'll see how well we can predict the rating based on the data about the wine (including the description). I found this on <kaggle.com>

```r
# clear the environment, just to be safe
rm(list=ls())

library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(tidyverse)
```

```
## Loading tidyverse: tibble
## Loading tidyverse: tidyr
## Loading tidyverse: readr
## Loading tidyverse: purrr
## Loading tidyverse: dplyr
```

```
## Conflicts with tidy packages ----------------------------------------------
```

```
## filter(): dplyr, stats
## lag():    dplyr, stats
## lift():   purrr, caret
```

```r
library(tidytext)
library(topicmodels)
require(janitor)
```

```
## Loading required package: janitor
```

```r
require(rio)
```

```
## Loading required package: rio
```

```r
wine <- rio::import("files/winemag-data_first150k.csv",
                    setclass = "tbl_df") %>%
  janitor::clean_names() %>%
  rename(id = v1)
```

We'll limit ourselves to a sample of 1000 observations for time's sake.

```r
#set.seed(227)

#wine <- sample_n(wine, 1000)
```

Let's take a look at the data

```r
wine
```

```
## # A tibble: 150,930 x 11
##       id country
##    <int>   <chr>
##  1     0      US
##  2     1   Spain
##  3     2      US
##  4     3      US
##  5     4  France
##  6     5   Spain
##  7     6   Spain
##  8     7   Spain
##  9     8      US
## 10     9      US
## # ... with 150,920 more rows, and 9 more variables: description <chr>,
## #   designation <chr>, points <int>, price <dbl>, province <chr>,
## #   region_1 <chr>, region_2 <chr>, variety <chr>, winery <chr>
```
So you can see wee have some information about where the wine is from, its rating (called `points`), its price called `price`, and textual description of the wine (called `description`). Let's see if we can train a model that does a good job predicting wine ratings.

### Extracting features: a brief detour into text analysis

Okay, before we start down the modelling road, we want to do something with the textual descriptions. Along these lines, we'll separate out the description, do some automated text analysis, and use the output of those analyses as features. This should make a bit more sense as we walk through.

First we need to load in a dataframe of stopwords these are basically words that don't have content and that we don't need (e.g., "the", "a", "and", etc.). Luckily, the R package `tidytext` has some built-in data on stop words. We need to load these up with the `data()` command.

```r
data("stop_words")
```

Now that we have the stop words, let's take our wine data set, select just the id (for merging purposes) and the text descriptions. Then we'll use the `unnest_tokens` function, which basically takes the descriptions, separates them by 'tokens' (which in this case is each word), and leaves us with a dataset with a row per word in each desciption (essentially an id X word from description lengthed dataset).

Finally, we'll use `anti_join()` to remove the stop words. This takes requires two dataframes as its two arguments, and removes any rows from the dataframe in the first argument that are in the dataframe in the second argument. Since we're using pipes (`%>%`), the first argument is invisible, but is the new expanded wine description data, and the second is the dataframe of stopwords; effectively, this will just remove the stopwords from our expanded wine description data.


```r
wine_text_expand<- wine %>%
  # take just the id and description
  select(id, description) %>%
  # unnest tokens; provide it the new variable name (for the tokens)
  # and the old variable (where it can find the text to tokenize).
  unnest_tokens(word, description) %>%
  anti_join(stop_words)
```

```
## Joining, by = "word"
```

Okay, now that we have  cleaned up text data for the wine descriptions, let's extract some features from the descriptions. We can use sentiment analysis. since that seems like it will definitely be relevant. Sentiment analysis is intended to extract the emotional tone of a text, and in this case, will basically give us a score corresponding to how poisitive and negative each word is. We'll leave it at sentiment analysis for the sake of time.

We'll do sentiment analysis using tidytext, and the "afinn" sentiment dictionary. This dictionary has a set of words with a continuous sentiment score (from -3, to +3; neutral point of 0). We can use it to get sentiment scores by using `inner_join()`, which basically keeps all columns, but only rows shared by the two dataframes; in this case, only words that are in both our descriptions data AND the sentiment dictionary will be kept, and columns for id, word, and sentiment score. Then, we'll summarize across words to get a sentiment score for each wine's description (wine being tracked with id). This will leave us with a dataframe with id and sentiment score (since the words are shared between the two dataframes)

```r
description_sentiment <- wine_text_expand %>%
  # this makes it so that all is saved is a
  # data frame that contains the words in the afinn
  # sentiment dictionary, the score associated with those words,
  # and the id for the wine.
  inner_join(get_sentiments("afinn")) %>% 
  # group by wine id
  group_by(id) %>% 
  # summarize such that we have a single sentiment score 
  # per wine id
  summarize(sentiment = mean(score)) 
```

```
## Joining, by = "word"
```

```r
description_sentiment
```

```
## # A tibble: 123,785 x 2
##       id sentiment
##    <int>     <dbl>
##  1     0       1.8
##  2     1       1.5
##  3     3       2.0
##  4     4       2.0
##  5     5       2.5
##  6     6       2.5
##  7     7       2.0
##  8     9       0.0
##  9    10      -1.0
## 10    12       4.0
## # ... with 123,775 more rows
```

And, let's merge that back into the wine dataframe.

```r
wine_for_ml <- wine %>%
  left_join(description_sentiment, by = "id") %>%
  # removing raw description for now
  select(points, price, sentiment) %>%
  # just removing missing values, because they complicate things
  na.omit()
```

### Modeling with caret

We have quite a bit of data here (150000 cases), so first let's partition our data into a training and test dataframe. We'll do a 75-25 training-test split, and can use caret's `createDataPartition()` function to do it.

```r
# Set seed for consistency's sake
set.seed(227)
# This part creates a list of values;
# these values are the row numbers for data included in the training set
# we're splitting it 75-25, such that 75% of cases will be in the training set (25% in the test).
in_train <- createDataPartition(y = wine_for_ml$points,
                                 p = .75,
                                 list = FALSE)
# subsets the training data (those data whose row number appears in the inTrain object)
training <- wine_for_ml[in_train,]
# subsets the test data (those data whose row number DOES NOT appear in the inTrain object)
testing <- wine_for_ml[-in_train,]
```

Okay, now that we have our training data, let's actually train a model.

First, we set up the training parameters using the `trainControl()` function. This is where you specify the method (e.g., cross-validation), and some other specifics.

In this example, we'll tell it to use 10-fold cross-validation, by specifying cross-validation as the method, and 10 as the number (i.e., k). We'll also tell it to save the best fitting model with the argument `savePredictions = TRUE`. You'll notice that we'll use these same controls for the different models we try (ridge, lasso, elastic net)

```r
# Sets parameters for training;
# telling it to use 10-fold cross-validation, and to save the predictions.
train_control<- trainControl(method="cv", number=10, 
                             savePredictions = TRUE)
```

#### Ridge Method
Now let's train a model using ridge regression. We do this by telling it to:

1) predict points from everything (including interactions). This is accomplished with .*. where '.' = all (well, all except the outcome variable).
2) using the training data
3) using the training parameters we just set
4) using the ridge method
5) pre-processing by centering and scaling (essentially z scoring everything; this is critical, because we want everything on the same scale, since parameter size is being penalized in one way or another).


```r
fit_ridge <- train(points ~ .*., 
                   data = training,
                   trControl = train_control,
                   method = "ridge",
                   preProc = c("center", "scale"))
```

```
## Loading required package: elasticnet
```

```
## Loading required package: lars
```

```
## Loaded lars 1.2
```

Okay, so how did our model do? We can evaluate this in two different ways given what we've done so far.

1) Easier test: what is the average fit (with $R^2$) and misfit (with $RMSE$) fromt the training. This will basically take the $R^2$ and $RMSE$ from all 10 folds and average them, telling us how well our models were doing on average across training runs.

We can get this information like so:

```r
avg_r2_ridge <- mean(fit_ridge$results$Rsquared)
avg_RMSE_ridge <- mean(fit_ridge$results$RMSE)

avg_r2_ridge
```

```
## [1] 0.2686071
```

```r
avg_RMSE_ridge
```

```
## [1] 2.759382
```

Okay, so an $\bar{R^2}$ of 0.27, meaning we are explaining 26.86% of the variance in wine ratings with sentiment and price (and the interaction).

2) Harder test: how well does it do with the holdout sample?


```r
pred_ridge <- predict(fit_ridge, newdata = testing)

# Gets R^2 and RMSE for ridge model
fitstat_ridge <- postResample(pred = pred_ridge, 
                                  obs = testing$points)
fitstat_ridge
```

```
##      RMSE  Rsquared 
## 2.8348519 0.2411339
```
Okay, so an $R^2$ of 0.24, meaning we are explaining 24.11% of the variance in wine ratings with sentiment and price (and the interaction).

#### Lasso Method

Okay, now let's try lasso. We'll use the same training-test split, and the same training parameters.

We will run virtually the same code, but change `method = "ridge"` to `method = "lasso"`, like so:

```r
fit_lasso <- train(points ~ .*., 
                   data = training,
                   trControl = train_control,
                   method = "lasso",
                   preProc = c("center", "scale"))
```

And let's evaluate this model in the same two ways. 

Easier test: average fit across training runs:

```r
avg_r2_lasso <- mean(fit_lasso$results$Rsquared)
avg_RMSE_lasso <- mean(fit_lasso$results$RMSE)

avg_r2_lasso
```

```
## [1] 0.249744
```

```r
avg_RMSE_lasso
```

```
## [1] 2.908199
```
Okay, so an $\bar{R^2}$ of 0.25, meaning we are explaining 24.97% of the variance in wine ratings with sentiment and price (and the interaction).

Now for the harder test.

Harder test: how does it do with the holdout sample?


```r
pred_lasso <- predict(fit_lasso, newdata = testing)

# Gets R^2 and RMSE for lasso model
fitstat_lasso <- postResample(pred = pred_lasso, 
                                  obs = testing$points)

fitstat_lasso
```

```
##      RMSE  Rsquared 
## 2.8329419 0.2400375
```

Okay, so an $R^2$ of 0.24, meaning we are explaining 24% of the variance in wine ratings with sentiment and price (and the interaction).

#### Elastic Net Method

And finally, let's do the same with elastic net. Like before, we'll use the same data split and training parameters. And again, the code is *almost* identical; we just change `method = "lasso"` to `method = "ridge"`


```r
fit_enet <- train(points ~ .*., 
                   data = training,
                   trControl = train_control,
                   method = "enet",
                   preProc = c("center", "scale"))
```

And let's evaluate the model in the same two ways.

Easier test: average fit across training runs


```r
avg_r2_enet <- mean(fit_enet$results$Rsquared)
avg_RMSE_enet <- mean(fit_enet$results$RMSE)

avg_r2_enet
```

```
## [1] 0.2480728
```

```r
avg_RMSE_enet
```

```
## [1] 2.923794
```
Okay, so an $\bar{R^2}$ of 0.25, meaning we are explaining 24.97% of the variance (on average) in wine ratings with sentiment and price (and the interaction).


Harder test: how does it do with the holdout sample?


```r
pred_enet <- predict(fit_enet, newdata = testing)
```

```
## Loading required package: elasticnet
```

```
## Loading required package: lars
```

```
## Loaded lars 1.2
```

```r
# Gets R^2 and RMSE for enet model
fitstat_enet <- postResample(pred = pred_enet, 
                                  obs = testing$points)
fitstat_enet
```

```
##      RMSE  Rsquared 
## 2.8348519 0.2411339
```

Okay, so an $R^2$ of 0.24, meaning we are explaining 24.11% of the variance in wine ratings with sentiment and price (and the interaction).

# Closing thoughts

I want to mention a few things in closing. The first is that you'll notice the three methods we tried in our example produced nearly identical fits. One reason for this is that we supplied a very small number of predictors (just 2 + an interaction, so 3 parameters). When you have many more predictors, these methods may start to differ a bit more (especially if the predictors are correlated, as we went over in the difference between ridge and lasso).

Finally, we split the data once into a training and test set. This is generally OK, BUT, if you're using the holdout sample to evaluate models (like we did here), you wouldn't want to use it to CHOOSE a model. That is, if we actually wanted to decide which of the three methods we wanted to use, the most defensible case would be to split the data into three sets: 

1) model training
2) model selection
3) test / model evaluation

This would keep the sort of purity of our test (model evaluation) data, and provide a good defense against overfitting.

# References:

Yarkoni, T., & Westfall, J. (2017). Choosing prediction over explanation in psychology: Lessons from machine learning. *Perspectives on Psychological Science*, 12(6), 1100-1122.

Kuhn, M., & Johnson, K. (2013). Applied predictive modeling. New York, NY: Spring-Verlag.
