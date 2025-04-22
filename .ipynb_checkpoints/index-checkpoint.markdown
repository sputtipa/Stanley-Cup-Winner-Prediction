---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

Hello Guys!

As I recently started watching NHL hockey, I have been very curious about how much of the regular season play and statistics indicate success in the postseason or the Stanley Cup Playoffs.

In this project, I will dive through how the regular season statistics correlate to the postseason results, as well as predict the winner of the 2025 Stanley Cup, as at the time I am reporting this result (April 20th, 2025), the playoffs have just begun.


# **Introduction**

First, I will explain how the NHL playoff format works. Every team plays 82 games in the regular season, and 16 teams will qualify for the playoffs, where the top 3 teams from each division (Atlantic, Metropolitan, Central, and Pacific Divisions) will qualify first. The last 4 spots are the 2 wild card teams from the East (Atlantic, Metropolitan) and West divisions (Central and Pacific Divisions). The wild card teams are the teams with the most points outside of the top 3 teams in the division. Each round is competed in a best-of-7 format, and the winners advance to the next round. You will have to win 4 rounds to win the Stanley Cup!

With that in mind, let's look at our dataset.

The dataset we will be working with is from [Money Puck](https://moneypuck.com/), which provides many regular season statistics of every team from each season.

As you can see from the Money Puck website, there are four big datasets from each season (year): Skaters, Goalie, Line, and Team Level. In this project, I chose to use only the team dataset, as it already contains more than 100 columns about the team’s regular season statistics.

#### **Understading The Dataset**

Initially, we have 16 datasets containing the team statistics from each regualar season (since 2009 to 2024). Keep in mind that I am only using data up to the 2023-2024 regualar season as I will be predicting the winner for this 2024-2025 year.

Within each raw dataset, it consists of 107 columns shown below:

<div style="margin-left: 20px">

<details>
<summary>Team & Game Info</summary>
<ul>
  <li>team</li>
  <li>season</li>
  <li>name</li>
  <li>team.1</li>
  <li>position</li>
  <li>games_played</li>
</ul>
</details>
<div style="margin-top: 5px;"></div>
<details>
<summary> Goals & xGoals</summary>
<ul>
  <li>goalsFor</li>
  <li>goalsAgainst</li>
  <li>xGoalsFor</li>
  <li>xGoalsAgainst</li>
  <li>reboundGoalsFor</li>
  <li>reboundGoalsAgainst</li>
  <li>reboundxGoalsFor</li>
  <li>reboundxGoalsAgainst</li>
  <li>xGoalsFromxReboundsOfShotsFor</li>
  <li>xGoalsFromActualReboundsOfShotsFor</li>
</ul>
</details>
<div style="margin-top: 5px;"></div>
<details>
<summary> Shot Attempts</summary>
<ul>
  <li>shotAttemptsFor</li>
  <li>shotAttemptsAgainst</li>
  <li>blockedShotAttemptsFor</li>
  <li>blockedShotAttemptsAgainst</li>
  <li>unblockedShotAttemptsFor</li>
  <li>unblockedShotAttemptsAgainst</li>
  <li>scoreAdjustedShotsAttemptsFor</li>
  <li>scoreAdjustedShotsAttemptsAgainst</li>
</ul>
</details>
<div style="margin-top: 5px;"></div>
<details>
<summary> Shot Danger Zones</summary>
<ul>
  <li>highDangerShotsFor</li>
  <li>highDangerShotsAgainst</li>
  <li>mediumDangerShotsFor</li>
  <li>mediumDangerShotsAgainst</li>
  <li>lowDangerShotsFor</li>
  <li>lowDangerShotsAgainst</li>
  <li>highDangerGoalsFor</li>
  <li>highDangerGoalsAgainst</li>
  <li>mediumDangerGoalsFor</li>
  <li>mediumDangerGoalsAgainst</li>
  <li>lowDangerGoalsFor</li>
  <li>lowDangerGoalsAgainst</li>
  <li>mediumDangerxGoalsFor</li>
  <li>mediumDangerxGoalsAgainst</li>
  <li>highDangerxGoalsFor</li>
  <li>highDangerxGoalsAgainst</li>
  <li>lowDangerxGoalsFor</li>
  <li>lowDangerxGoalsAgainst</li>
</ul>
</details>
<div style="margin-top: 5px;"></div>
<details>
<summary> Zone Transition Events</summary>
<ul>
  <li>xPlayStoppedFor</li>
  <li>xPlayStoppedAgainst</li>
  <li>xPlayContinuedInZoneFor</li>
  <li>xPlayContinuedInZoneAgainst</li>
  <li>xPlayContinuedOutsideZoneFor</li>
  <li>xPlayContinuedOutsideZoneAgainst</li>
  <li>playStoppedFor</li>
  <li>playStoppedAgainst</li>
  <li>playContinuedInZoneFor</li>
  <li>playContinuedInZoneAgainst</li>
  <li>playContinuedOutsideZoneFor</li>
  <li>playContinuedOutsideZoneAgainst</li>
  <li>zoneGiveawaysFor</li>
  <li>zoneGiveawaysAgainst</li>
</ul>
</details>
<div style="margin-top: 5px;"></div>
<details>
<summary> Events & Discipline</summary>
<ul>
  <li>faceOffsWonFor</li>
  <li>faceOffsWonAgainst</li>
  <li>penaltiesFor</li>
  <li>penaltiesAgainst</li>
  <li>penaltyMinutesFor</li>
  <li>penaltyMinutesAgainst</li>
  <li>hitsFor</li>
  <li>hitsAgainst</li>
  <li>takeawaysFor</li>
  <li>takeawaysAgainst</li>
  <li>giveawaysFor</li>
  <li>giveawaysAgainst</li>
</ul>
</details>
<div style="margin-top: 5px;"></div>
<details>
<summary> Goalie Metrics</summary>
<ul>
  <li>savedShotsOnGoalFor</li>
  <li>savedShotsOnGoalAgainst</li>
  <li>savedUnblockedShotAttemptsFor</li>
  <li>savedUnblockedShotAttemptsAgainst</li>
  <li>missedShotsFor</li>
  <li>missedShotsAgainst</li>
  <li>xFreezeFor</li>
  <li>xFreezeAgainst</li>
  <li>freezeFor</li>
  <li>freezeAgainst</li>
</ul>
</details>
<div style="margin-top: 5px;"></div>
<details>
<summary> Adjusted Metrics</summary>
<ul>
  <li>flurryAdjustedxGoalsFor</li>
  <li>flurryAdjustedxGoalsAgainst</li>
  <li>flurryScoreVenueAdjustedxGoalsFor</li>
  <li>flurryScoreVenueAdjustedxGoalsAgainst</li>
  <li>scoreVenueAdjustedxGoalsFor</li>
  <li>scoreVenueAdjustedxGoalsAgainst</li>
  <li>scoreAdjustedTotalShotCreditFor</li>
  <li>scoreAdjustedTotalShotCreditAgainst</li>
  <li>scoreFlurryAdjustedTotalShotCreditFor</li>
  <li>scoreFlurryAdjustedTotalShotCreditAgainst</li>
  <li>totalShotCreditFor</li>
  <li>totalShotCreditAgainst</li>
</ul>
</details>
<div style="margin-top: 5px;"></div>
<details>
<summary> Extra xG Events</summary>
<ul>
  <li>xOnGoalFor</li>
  <li>xOnGoalAgainst</li>
  <li>xReboundsFor</li>
  <li>xReboundsAgainst</li>
  <li>xReboundxGoalsFor</li>
  <li>xReboundxGoalsAgainst</li>
</ul>
</details>
<br>
</div>

As you can see, there's so many columns we can work with, but these are only the column we should know what it means:

<div style="margin-left: 20px">

<strong>xGoalsFor:</strong> Measures expected goals scored — strong offensive indicator.
<div style="margin-top: 5px;"></div>

<strong>xGoalsAgainst:</strong> Expected goals conceded — captures defensive strength.
<div style="margin-top: 5px;"></div>

<strong>goalsFor:</strong> Actual goals scored — directly tied to team success.
<div style="margin-top: 5px;"></div>

<strong>goalsAgainst:</strong> Actual goals allowed — key for defensive efficiency.
<div style="margin-top: 5px;"></div>

<strong>scoreAdjustedShotsAttemptsFor:</strong> Measures puck possession adjusted for game context.
<div style="margin-top: 5px;"></div>

<strong>highDangerShotsFor:</strong> Captures the number of high-quality scoring chances.
<div style="margin-top: 5px;"></div>

<strong>highDangerShotsAgainst:</strong> Opponent's dangerous chances — key to defensive performance.
<div style="margin-top: 5px;"></div>

<strong>reboundxGoalsFor:</strong> Captures second-chance scoring opportunities.
<div style="margin-top: 5px;"></div>

<strong>saveFreezeFor / xFreezeFor:</strong> Goalie’s ability to control rebounds and play pace.
<div style="margin-top: 5px;"></div>

<strong>faceOffsWonFor:</strong> Possession control — especially important in special teams.
<div style="margin-top: 5px;"></div>

<strong>penaltyMinutesAgainst:</strong> Discipline — fewer penalties = less time shorthanded.
<div style="margin-top: 5px;"></div>

<strong>scoreVenueAdjustedxGoalsFor:</strong> Adjusted for home/away and game state — very predictive.

</div>
<br>

Each dataset contain 5 x number of team in the NHL that year as each team will have the correspoding value to each situation of other, all, 5on5, 4on5, and 5on4. For simplicity we will only use the 5on4 row which shorten our dataset to only around 32 rows per season (32 team in the NHL)

#### **Why this dataset?**

I will be using these value from each column corresponding to each team to predict the win-lose percentage of the team that make it to the playoff in the post season.

So where's that data? -- Great, Queations! I have scraped all the Playoffs win lose percentage from [Hockey Reference](https://www.hockey-reference.com/) and store it as a seperate dataset for each year.

---
<div style="margin-top: 10px;"></div>

# **Data Cleaning and Exploratory Data Analysis**

Now it's time to take a deep dive into our data and see if there's anything interesting about it. However, we need to clean up our data first!

#### **Data Cleaning**

Right now, we have two dictionaries, each consisting of a DataFrame for every season from 2009 to 2024. The first dictionary, team, contains the regular season statistics, and the second one, playoffs, contains the playoff results.

First, I needed to standardize all the team names in these DataFrames by converting them to their official abbreviation format. For example, both "Anaheim Ducks" and "Mighty Ducks of Anaheim" were changed to "ANA".

After that, I removed some duplicate and irrelevant columns such as name, team.1, position, games_played, and iceTime to make the DataFrame easier to work with. Then, I converted all columns that had a data type other than float into float for easier analysis.

Since we want to explore how regular season statistics correlate with postseason win-loss percentage, I merged the team DataFrame with the playoff DataFrame. I filtered out all teams that didn’t make the playoffs and created a new DataFrame combining all seasons. This final DataFrame includes only the teams that made the playoffs each season, along with their corresponding playoff win-loss percentage in the last column.

Finally, the combined DataFrame had two columns with missing values due to typos in some of the original season DataFrames, where 'penaltiesFor' and 'penaltiesAgainst' were misspelled as 'penalitiesFor' and 'penalitiesAgainst'. I dropped all of these columns, as they represent the number of penalties for and against — which we already capture through 'penaltyMinutesFor' and 'penaltyMinutesAgainst', making them interchangeable for our purposes.

Here is the **Cleaned DataFrame Head**:

<iframe src="df_head2.html" width="120%" height="300px" frameborder="0"></iframe>
<br>

#### **Univariate Analysis**

After we have a good dataframe to work with, let's see the trend of these feature.

First, let's understand the ditribution of the value we will be predicting, the win lose percentage of each team in the playoff. 

<iframe src="playoff_win_dist.html" width="120%" height="400px" frameborder="0"></iframe>

This plot shows the distribution of playoff win-loss percentage from 2009 to 2024 for all teams that made the postseason. Values are range between 0 and 1, where 1 represents teams that won every game and 0 represents teams that were swept. Most teams fall between 0.4 and 0.6, showing how competitive the playoffs are and how rare it is for teams to completely dominate or be eliminated without a win.

Some other notable feature distribution are the distributions of goalsFor and goalsAgainst in the regular season. 

<iframe src="goalsFor_dist.html" width="120%" height="400px" frameborder="0"></iframe>
<iframe src="goalsAgainst_dist.html" width="120%" height="400px" frameborder="0"></iframe>

This show that most playoff teams score between 210 and 280 goals while conceding between 200 and 240 which highlights that while elite offense is a common trait among playoff teams, postseason qualification often depends more on maintaining a strong goal differential than having an exceptionally low goals-against total.
<br>

#### **Bivariate Analysis**

Now that we’ve seen the distribution of the value we are predicting — the playoff win-loss percentage — and examined some key features, we can visualize how each of these regular season statistics corresponds to playoff performance.

<iframe src="2-goalsFor.html" width="120%" height="400px" frameborder="0"></iframe>
<iframe src="2-goalsAgainst.html" width="120%" height="400px" frameborder="0"></iframe>
<iframe src="2-highDangerShotsFor.html" width="120%" height="400px" frameborder="0"></iframe>
<iframe src="2-xFreezeFor.html" width="120%" height="400px" frameborder="0"></iframe>

The scatter plots above show relationships between several features and playoff success. While teams with higher goalsFor tend to have slightly higher win percentages, and teams with lower goalsAgainst seem to cluster higher, there is no strong linear correlation in any single feature. Metrics like highDangerShotsFor, and xFreezeFor also show considerable spread across all win percentages. This shows that while these stats may contribute contextually, no individual variable strongly determines playoff outcomes on its own.

These trends support the idea that playoff success is multifactorial — and likely depends on a combination of stats rather than any one dominant regular season metric.

#### **Interesting Aggregates**

<iframe src="playoff_pivot_table.html" width="80%" height="168x" frameborder="0"></iframe>

This pivot table groups teams by their playoff win percentage and shows the average of several regular season stats. It shows that teams with higher playoff success tend to have more goalsFor and fewer goalsAgainst, suggesting that both offensive production and defensive reliability in the regular season are tied to postseason outcomes.

---
<div style="margin-top: 10px;"></div>
# **Framing a Prediction Problem**

#### **Prediction Problem Statement**

The goal of this project is to predict how well a team will perform in the playoffs (measured by playoff win-loss percentage) based on its regular season statistics. Specifically, I want to explore:  
<div style="margin-top: 10px;"></div>
<div style="margin-left: 30px"><em>What aspects of a team's regular season performance are most important in determining playoff success, and do these features follow a clear trend — or are they too inconsistent to identify any strong pattern?</em></div>
<div style="margin-top: 10px;"></div> 

This is a regression problem, not classification, because the target variable "playoff_win_percentage" is a continuous value between 0 and 1, rather than a discrete category like "win" or "lose".

#### **Evaluation Metric**

I am using both Mean Squared Error (MSE) and Mean Absolute Error (MAE) to evaluate model performance.

- **MSE**: as a lot of models and methods only support minimizing this value
- **MAE**: is easy to interpret and robust to outliers, which gives a clear sense of the average error in win percentage.

I chose MSE and MAE over accuracy-style metrics as this is a regression task, not classification and I want to understand how far off my predictions are numerically — not just if the direction is right.

#### **Trends in the Data**

Several features (like goalsFor and shotAttemptsFor) show a loose positive relationship with playoff success.

However, these trends are not strongly linear, which suggests that there may be nonlinear relationships or interactions between features. It might also just be that luck plays a large role in playoff outcomes, or teams are simply playing differently in the playoffs.

This is why I will be trying more flexible models beyond simple linear regression.

---
<div style="margin-top: 10px;"></div>
# **Baseline Model**

The response variable is: playoff_win_percentage which is the (number of playoff wins / total playoff games).

This was chosen because it reflects a team’s overall performance in the postseason regardless of how far they advanced (e.g., some teams go 4–3 in Round 1, others 16–4 across 4 rounds).

I selected regular season statistics from teams that made the playoffs between 2009 and 2024. After data cleaning and merging, the features were all quantitative (e.g., goalsFor, xGoalsAgainst, blockedShotAttemptsFor, etc.). I did not pick any of the features based on their meaning, as from the analysis before, no feature indicated a strong linear relationship with playoff win percentage. Instead, I used underlying correlation to help me pick the features.

#### **Model Selection and Performance Summary**

To predict playoff win-loss percentage, I trained and compared multiple regression models including:

- ElasticNet
- k-Nearest Neighbors
- Random Forest
- MLP Neural Network

I chose to try these four models because the relationship between regular season statistics and playoff win-loss percentage is not strictly linear. Visualizations and scatter plots showed a wide variance in playoff outcomes for teams with similar regular season metrics. Therefore, I included a mix of model types: ElasticNet as a linear baseline for comparison, as it incorporates both L1 and L2 regularization; k-Nearest Neighbors and Random Forest to capture localized and ensemble-based nonlinear patterns; and Neural Network (MLPRegressor) for its potential to model highly nonlinear interactions.

Each model was wrapped in a Pipeline, and I applied StandardScaler() to normalize input features for models sensitive to scale, such as KNN and the neural network.

Here is the MAE and MSE results:

<iframe src="model4_results.html" width="80%" height="150x" frameborder="0"></iframe>

From the results, ElasticNet had the lowest MSE, while the Neural Network had the highest MSE on the testing data.

This result is interesting because I initially expected the non-linear models—such as k-Nearest Neighbors, Random Forest, and especially Neural Networks—to perform better based on earlier analysis and scatter plot visualizations. So, I wasn't fully convinced that ElasticNet, a linear model, could capture the complex relationships in the data. Although it performed surprisingly well (with an MAE of 0.1256), the result may be misleading.

To explore further, I used GridSearchCV to test the optimal number of features for each model, ranging from 5 to using all available columns. This helped me better understand how each model performs with different feature set sizes and helped avoid adding unnecessary complexity to the pipeline.

<iframe src="hello-chart.html" width="80%" height="150x" frameborder="0"></iframe>
ElasticNet used only five features for the best results, but when tested earlier with over 100 features, it still gave similar results. This is concerning, as overfitting may occur or the inclusion of highly correlated and redundant features could hurt performance.

In contrast, k-Nearest Neighbors and Random Forest stayed relatively stable across trials, showing they are more robust to different feature selections. As expected, the Neural Network (MLPRegressor) showed a big improvement. This improvement supports the idea that neural networks can do better when paired with the right features and hyperparameters. It also shows that these models can find patterns in complex, high-dimensional data that simpler models might miss.

Based on this, I chose the MLPRegressor (Neural Network) as my baseline model because:
- It can model nonlinear relationships between the predictors and playoff success.
- It didn't perform well at first, but I believed it had the most potential to improve with better feature selection and scaling.
- It benefits a lot from hyperparameter tuning, which I will focus on next.

The features selected by SelectKBest under cross-validation, since I have limited data, are: xGoalsPercentage, corsiPercentage, fenwickPercentage, freezeFor, and playContinuedInZoneAgainst.

The model’s current MAE is 0.144867 and MSE is 0.030148, which is better than I expected. Based on the earlier analysis, there isn’t a strong correlation between individual features and playoff win-loss percentage. However, with more fine-tuning, I think I can improve the model even more.

---
<div style="margin-top: 10px;"></div>
# **Final Model**

To improve upon the baseline regression model, I engineered additional features, refined the input space using feature selection, and tuned a neural network model to better capture the complex patterns within the data.


#### **Model Selection Process**
To avoid overfitting and ensure generalization, I selected the top 25 features using f_regression F-scores and then randomly sampled 1,000 combinations of 5 features from this pool. For each combination, I ran a full grid search using GridSearchCV over MLP hyperparameters and selected the configuration with the lowest test MSE. 

This approach differs from the method used in Part 4, where I relied on SelectKBest with cross-validation to identify top features. 

In contrast, the Part 5 strategy is more brute-force and correlation-driven for a broader exploration of potential feature interactions and redundancies. This method helped gathers feature sets that may not have ranked highest individually but worked well together in combination, ultimately leading to improved model performance.

This strategy allowed for wide exploration of feature interactions and model configurations without overfitting to a single full-featured model.

The 5 features showcasing the best combination are

- xGoalsPercentage
- goalsAgainst
- lowDangerxGoalsAgainst
- scoreAdjustedTotalShotCreditAgainst
- reboundxGoalsAgainst

This list actually shows some very insightful findings and makes a lot of sense beyond just randomly choosing some features. This is because we can see that all of the features are related to goals. However, 4 out of the 5 are about goals against, which shows that defensive skills have a very strong impact on winning.

#### **Feature Engineering**

I created two new features based on the best 5 features identified during the GridSearchCV and random sampling of feature combinations, where I chose to focus on reboundxGoalsAgainst because whenever I watch hockey, those kinds of goals can significantly help win games:

Log_ReboundxGoalsAgainst
- Why? The original feature xReboundxGoalsAgainstt likely contains a right-skewed distribution with potential outliers. Applying a logarithmic transformation (log1p) helps stabilize variance and reduce skewness which is easier for the neural network to learn meaningful patterns. This transformation also shows that after a certain point, more rebound goals don’t have as much added impact, they start to matter less.

Scaled_ReboundxGoalsAgainst
- Why? This is the same rebound-related feature but explicitly scaled using StandardScaler. While the variable is already informative, scaling it provides finer control over its influence during training. Since rebound goals often result from defensive breakdowns or second-chance opportunities, emphasizing its normalized magnitude can help the model better recognize patterns of defensive inefficiency. The scaled version complements the logarithmic one by preserving relative magnitudes and helping with convergence during optimization.

The final input features to the model were:

- xGoalsPercentage
- goalsAgainst
- lowDangerxGoalsAgainst
- scoreAdjustedTotalShotCreditAgainst
- reboundxGoalsAgainst
- Log_xreboundxGoalsAgainst (engineered)
- Scaled_xreboundxGoalsAgainst (engineered)

#### **Model Choice: MLP Neural Network**
I selected a Multi-Layer Perceptron (MLPRegressor) as the final model due to its ability to model complex, nonlinear relationships between regular season metrics and playoff outcomes.

After tuning via GridSearchCV, the optimal hyperparameters were:
- hidden_layer_sizes: (128,)
- alpha: 0.0001
- max_iter: 6000
- early_stopping: True

The choice of a single layer with 128 units likely balanced expressiveness with regularization, while the low alpha (regularization strength) avoided underfitting.

#### **Final Model Performance**

- Final MAE: 0.1180
- Final MSE: 0.0232

Compared to the baseline:
- Baseline MLP MAE: 0.140867
- Baseline MLP MSE: 0.030148

The final model has a 16.27% improvement in MAE and a 23.01% reduction in MSE, showing that feature engineering, selection, and hyperparameter tuning substantially improved performance.

I will evaluate my final model based on the MAE, as it shows the true decimal points of how wrong my prediction is. Since we are predicting the playoff win-loss percentage, every team will have a very similar final value, so every decimal point is important.

So, to answer the question we’ve been doing all of this for: does regular season performance translate to playoff success?

First, take a look at the predicted percentage for each team in the 2025 season based on our final model. I would say the chart results actually look very realistic (comparing them to my picks), which may not be how this should be judged, but we can see that the model didn’t just pick the team with the highest points in the regular season. It used other features we discussed.

<iframe src="dzo.html" width="80%" height="430x" frameborder="0"></iframe>

Even though the Colorado Avalanche (COL), ranked third, is my pick, I would say the majority of regular season statistics actually play a very small role in how the model performs. However, some particular stats — like how good a team is at defending against rebound goals — may actually be among the most important aspects of winning the Stanley Cup.

So we’ll see if this model is right — or if I am :-)
