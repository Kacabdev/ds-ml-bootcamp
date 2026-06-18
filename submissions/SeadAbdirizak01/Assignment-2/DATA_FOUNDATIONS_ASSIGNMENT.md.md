# Data Foundations for Machine Learning
## A Self-Collected Dataset on Football Highlight Video Engagement

**Prepared by:** SeadAbdirizak
**Assignment:** Data Foundations for Machine Learning (Lessons 1–3)
**Date:** June 2026

---

## 1. Title and Collection Method

**Dataset title:** *Engagement Performance of Football Highlight Clips on Social Media*

I work as a content editor producing short football highlight reels (Messi, Ronaldo, Neymar, Argentina, Barcelona, and a few local clubs), which I publish on TikTok, Instagram, and Facebook, some under my own page and a few under the Sead Creativity brand for clients. For this assignment I treated my own publishing history as a live data source instead of pulling something from Kaggle.

Between May 20 and June 14, 2026, I logged every highlight clip I edited and posted, recording how it was made and how it performed. The collection method was observation: each time I published a video, I noted production details (length, color grading, caption language, posting time) in a spreadsheet, then went back 7 days later to record the views, likes, comments, and shares shown on the platform. This gave me 55 rows, each one a single published video.

This connects directly to where data collection sits in the data science lifecycle: before any cleaning, modeling, or analysis can happen, someone has to define what's worth measuring and go get it. That's the stage this dataset belongs to. Lesson 4 will move it into preprocessing.

## 2. Description of Features and Label

The dataset has two kinds of columns: things I knew *before* publishing a video (the features, X), and things I could only know *after* publishing it (the raw outcome numbers, which I used to build the label, y).

**Features (X), known at publish time:**

| Feature | Type | Description |
|---|---|---|
| Platform | Categorical | TikTok, Instagram, or Facebook |
| Video Length (seconds) | Numeric | Clip duration |
| Player/Team Featured | Categorical | Messi, Ronaldo, Neymar, Argentina, Barcelona, Local Team, Other |
| Caption Language | Categorical | Somali, English, Arabic, or Mixed |
| Posting Time | Categorical | Morning, Afternoon, Evening, Night |
| Day of Week | Categorical | Monday through Sunday |
| Color Graded | Binary | Whether I applied color correction before export |
| Burned-in Captions | Binary | Whether subtitles were embedded in the video itself |

**Raw outcome data, recorded 7 days after posting:**

Views, Likes, Comments, Shares.

**Label (y):** From the raw outcomes I calculated an **Engagement Rate**, defined as (Likes + Comments + Shares) ÷ Views × 100. This single number is what the dataset is ultimately trying to explain or predict, and it's the column that turns this from a pile of numbers into a supervised learning problem.

## 3. Dataset Structure

The full dataset has **55 rows and 13 columns** (8 features, 4 raw outcome metrics, and 1 derived label). Below is a sample of 8 rows, showing the columns most relevant to the prediction problem.

| ID | Platform | Length (s) | Player/Team | Caption Lang | Posting Time | Color Graded | Views | Likes | Comments | Shares | Engagement Rate (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TikTok | 38 | Messi | Somali | Evening | Yes | 14,200 | 1,850 | 96 | 210 | 15.18 |
| 2 | Instagram | 52 | Ronaldo | English | Night | Yes | 9,800 | 740 | 41 | 88 | 8.88 |
| 3 | Facebook | 27 | Neymar | Mixed | Afternoon | No | 5,200 | 310 | 15 | 22 | 6.67 |
| 4 | TikTok | 45 | Argentina | Somali | Morning | Yes | 22,100 | 3,100 | 188 | 402 | 16.69 |
| 5 | Instagram | 33 | Barcelona | English | Evening | No | 4,100 | 205 | 9 | 14 | 5.56 |
| 6 | TikTok | 60 | Local Team | Somali | Afternoon | Yes | 1,800 | 95 | — | 7 | (missing) |
| 7 | Facebook | "0:41"* | Messi | English | Night | No | 6,300 | 410 | 19 | 31 | 7.33 |
| 8 | TikTok | 29 | Ronaldoo* | English | Night | Yes | 8,700 | 690 | 33 | 71 | 9.13 |

\* Row 7 was logged in mm:ss format instead of seconds. Row 8 has a typo in the team name ("Ronaldoo"). Both are kept in the table on purpose, to show what the raw data actually looked like before cleaning.

## 4. Quality Issues

The dataset is exactly as messy as a manually logged spreadsheet tends to be:

**Missing values.** 4 of the 55 rows are missing a comment count, mostly from early in the collection period before I started checking that field consistently. Engagement rate can't be calculated for these without first deciding how to handle the gap.

**Inconsistent formats.** 3 entries recorded video length as mm:ss ("0:41") instead of a plain number of seconds, because I copied the duration straight off the editing timeline in CapCut without converting it.

**Typos in categorical text.** Player and team names have inconsistent spelling in a few places, "Ronaldoo" instead of "Ronaldo," and "Barca" instead of "Barcelona" in two rows. Without cleaning, a model would treat these as separate categories.

**Duplicate entries.** Two videos appear twice in the log. I had recorded their numbers once at the 24-hour mark and again at 7 days, before settling on a single consistent measurement window, and forgot to delete the earlier rows.

**Class imbalance.** Messi, Ronaldo, and Neymar content together account for 36 of the 55 videos. Local Somali club content makes up only 4. Any model trained on this would learn a lot about global stars and very little about local teams.

**Outliers.** One clip (an Argentina World Cup compilation) pulled in over 40,000 views, roughly four times the average for the rest of the dataset, which will skew any model that isn't built to handle it.

## 5. Learning Type: Supervised

This is a **supervised learning** problem. Every row pairs a set of known inputs (platform, length, team, caption language, timing, editing choices) with a measurable output (engagement rate) recorded after the fact. That input-output pairing is the defining feature of supervised learning from Lesson 2: the model learns a mapping from X to y using examples where y is already known.

There's a secondary, unsupervised angle worth mentioning. If I dropped the engagement rate column entirely and just clustered the videos by their feature values (length, team, caption language, timing), I could look for natural groupings in *how* I produce content, independent of how well it performed. That would be a legitimate unsupervised exploration, but it's not the main framing here, since the dataset was built specifically around a label I can both calculate and care about.

## 6. Use Case and Lifecycle Placement

I want to use this data for two main things: first, predicting the exact engagement rate a video will get before I even publish it (regression), and second, just sorting upcoming clips into High, Medium, or Low engagement buckets so I know which ones deserve extra promotion (classification).

**Regression** works off the planned length, team, caption language, and posting time. A reasonably accurate model could tell me, before I even export a clip, whether it's worth posting at 9pm versus waiting for the evening slot, or whether a Somali caption will outperform an English one for a given matchup.

**Classification** is the simpler, more forgiving version of the same problem. I don't need an exact number, just a quick flag for which clips are worth pushing extra promotion behind.

In terms of the data science lifecycle from Lesson 1, this assignment covers the **data collection** stage. Lesson 4 will move it into **data preparation**: filling or dropping the missing comment counts, standardizing the length column to seconds, fixing the name typos, removing the duplicate rows, and encoding categorical columns like platform and caption language into numeric form so a model can actually use them. After that comes modeling and evaluation (training a regression or classification model and checking its error rate), and eventually deployment, which for me would realistically just mean a small checklist or script I run before publishing a new clip.

## Next Steps

The dataset itself (the full 55-row spreadsheet) can be cleaned up and exported as a CSV file if needed for submission alongside this report.
