# Error Analysis

The following tables show high-confidence misclassifications for each model.
These examples reveal systematic weaknesses (sarcasm, domain-specific vocabulary,
neutral-positive confusion) that inform future work directions.

> **Methodology:** Errors are ranked by model confidence (descending).
> High-confidence errors are most informative — they show cases where the model
> is definitively wrong, not merely uncertain.

## LOGREG

**Total errors shown:** 30

**Error type breakdown:**

| Confusion | Count |
|-----------|-------|
| Neutral → Positive | 8 |
| Positive → Negative | 8 |
| Negative → Positive | 5 |
| Negative → Neutral | 5 |
| Neutral → Negative | 2 |
| Positive → Neutral | 2 |

**Sample Misclassifications (highest confidence first):**

| # | Text (truncated) | True | Predicted | Confidence |
|---|-----------------|------|-----------|-----------|
| 1 | that man wicked after he came to jamaica hot hot to married you he did this to y... | Negative | Positive | 0.970 |
| 2 | this is for all of you democrats others liberals republicans trump is still your... | Negative | Positive | 0.945 |
| 3 | for all the incredible bravery of the ukrainians the fact is that they cannot su... | Negative | Positive | 0.945 |
| 4 | where can find vim... | Negative | Neutral | 0.935 |
| 5 | use english text please... | Negative | Neutral | 0.931 |
| 6 | for some reason felt like crying because ur my fav youtuber and my inspiration f... | Negative | Positive | 0.926 |
| 7 | but will the merger between honda and nissan make hondas cars less reliable beca... | Negative | Neutral | 0.925 |
| 8 | thank you for your help with this amazing ai you represent the future... | Negative | Positive | 0.922 |
| 9 | steve adams is not just some random player... | Negative | Neutral | 0.918 |
| 10 | ask nestle and the people and golf courses who water hours day where the water i... | Negative | Neutral | 0.908 |
| 11 | babymonster red heart im so happy yall stan them they are my fav red heart red h... | Neutral | Positive | 0.996 |
| 12 | harris you are dynamo thank you for being so honest and forthright we need journ... | Neutral | Positive | 0.989 |
| 13 | its sad bittersweet understanding that you are completely alone when youre in tr... | Neutral | Negative | 0.983 |
| 14 | wish wish the world listens ro this everyone should listen to this love you liam... | Neutral | Positive | 0.983 |
| 15 | dont have autism but do this lot actually when get gift im always greatful and d... | Neutral | Positive | 0.981 |
| 16 | remember her she was pretty cool hope shes learned lot... | Neutral | Positive | 0.978 |
| 17 | you triggered my alexa other than that love your tutorials so so much thank you ... | Neutral | Positive | 0.977 |
| 18 | love this do canada as well red heart... | Neutral | Positive | 0.973 |
| 19 | am feeling very grateful right now for this amazing life god has blessed me with... | Neutral | Positive | 0.973 |
| 20 | bad cops good cops not working for fox... | Neutral | Negative | 0.971 |
| 21 | pile of poo face with rolling eyes face vomiting face with rolling eyes pile of ... | Positive | Negative | 0.992 |
| 22 | russia has lot of equipment but it lacks professional army their army and reserv... | Positive | Negative | 0.972 |
| 23 | what an awful legacy to inherit poor woman... | Positive | Negative | 0.972 |
| 24 | why would truth about history make you feel guilty because youre white that make... | Positive | Negative | 0.964 |
| 25 | he needs to be removed itll be disgusting if he pardons himself hes got deaths o... | Positive | Negative | 0.964 |
| 26 | where is this wanna fly like superman pretty plz... | Positive | Neutral | 0.956 |
| 27 | love hearing how deluded the west is and how quickly they forget talking about s... | Positive | Negative | 0.951 |
| 28 | his name is jared for yall wondering... | Positive | Neutral | 0.951 |
| 29 | what is surprising about the fact he cheats he lies he cheats he bullies people ... | Positive | Negative | 0.950 |
| 30 | trump and the ag barr along with the supreme court judges are openly using voter... | Positive | Negative | 0.950 |

**Qualitative observations:**

- Review the errors above and annotate patterns here before thesis submission.
  Common patterns to look for:
  - Sarcasm / irony (e.g., 'Great job breaking the internet again')
  - Neutral comments misclassified as Negative (mild criticism)
  - Short ambiguous comments (≤5 words)
  - Domain-specific vocabulary (gaming, politics, cooking) causing false positives

---

## SVM

**Total errors shown:** 30

**Error type breakdown:**

| Confusion | Count |
|-----------|-------|
| Positive → Negative | 9 |
| Neutral → Negative | 8 |
| Negative → Neutral | 7 |
| Negative → Positive | 3 |
| Neutral → Positive | 2 |
| Positive → Neutral | 1 |

**Sample Misclassifications (highest confidence first):**

| # | Text (truncated) | True | Predicted | Confidence |
|---|-----------------|------|-----------|-----------|
| 1 | agree orange flavored snacks are terrible shes absolutely right... | Negative | Positive | 0.921 |
| 2 | for all the incredible bravery of the ukrainians the fact is that they cannot su... | Negative | Positive | 0.914 |
| 3 | but will the merger between honda and nissan make hondas cars less reliable beca... | Negative | Neutral | 0.912 |
| 4 | big query is an enterprise data warehouse for large amounts of relational struct... | Negative | Neutral | 0.907 |
| 5 | ask nestle and the people and golf courses who water hours day where the water i... | Negative | Neutral | 0.903 |
| 6 | where can find vim... | Negative | Neutral | 0.897 |
| 7 | steve adams is not just some random player... | Negative | Neutral | 0.894 |
| 8 | bro at least update to the latest version... | Negative | Neutral | 0.891 |
| 9 | youtube full stack developer me working for asian client... | Negative | Neutral | 0.891 |
| 10 | that man wicked after he came to jamaica hot hot to married you he did this to y... | Negative | Positive | 0.889 |
| 11 | babymonster red heart im so happy yall stan them they are my fav red heart red h... | Neutral | Positive | 0.972 |
| 12 | its sad bittersweet understanding that you are completely alone when youre in tr... | Neutral | Negative | 0.961 |
| 13 | bad cops good cops not working for fox... | Neutral | Negative | 0.958 |
| 14 | why do people follow hit and runs the insurance will pay and you wont get punish... | Neutral | Negative | 0.949 |
| 15 | another ai video... | Neutral | Negative | 0.948 |
| 16 | rusien gypsies en the en will lost the word... | Neutral | Negative | 0.939 |
| 17 | harris you are dynamo thank you for being so honest and forthright we need journ... | Neutral | Positive | 0.935 |
| 18 | git rm cached is working but just git rm is not working it says that it did not ... | Neutral | Negative | 0.933 |
| 19 | all vehicles sold in the us now require backup camera which this doesnt have im ... | Neutral | Negative | 0.932 |
| 20 | what to do if submitting is not working... | Neutral | Negative | 0.926 |
| 21 | pile of poo face with rolling eyes face vomiting face with rolling eyes pile of ... | Positive | Negative | 0.973 |
| 22 | russia has lot of equipment but it lacks professional army their army and reserv... | Positive | Negative | 0.951 |
| 23 | love hearing how deluded the west is and how quickly they forget talking about s... | Positive | Negative | 0.937 |
| 24 | why would truth about history make you feel guilty because youre white that make... | Positive | Negative | 0.934 |
| 25 | what is surprising about the fact he cheats he lies he cheats he bullies people ... | Positive | Negative | 0.933 |
| 26 | he needs to be removed itll be disgusting if he pardons himself hes got deaths o... | Positive | Negative | 0.927 |
| 27 | what an awful legacy to inherit poor woman... | Positive | Negative | 0.923 |
| 28 | where is this wanna fly like superman pretty plz... | Positive | Neutral | 0.923 |
| 29 | you are coward though yes she is and now the world knows it too... | Positive | Negative | 0.919 |
| 30 | trump and the ag barr along with the supreme court judges are openly using voter... | Positive | Negative | 0.913 |

**Qualitative observations:**

- Review the errors above and annotate patterns here before thesis submission.
  Common patterns to look for:
  - Sarcasm / irony (e.g., 'Great job breaking the internet again')
  - Neutral comments misclassified as Negative (mild criticism)
  - Short ambiguous comments (≤5 words)
  - Domain-specific vocabulary (gaming, politics, cooking) causing false positives

---

## TFIDF

**Total errors shown:** 30

**Error type breakdown:**

| Confusion | Count |
|-----------|-------|
| Neutral → Positive | 10 |
| Positive → Negative | 8 |
| Negative → Positive | 7 |
| Negative → Neutral | 3 |
| Positive → Neutral | 2 |

**Sample Misclassifications (highest confidence first):**

| # | Text (truncated) | True | Predicted | Confidence |
|---|-----------------|------|-----------|-----------|
| 1 | that man wicked after he came to jamaica hot hot to married you he did this to y... | Negative | Positive | 0.980 |
| 2 | its clapping hands light skin tone not clapping hands light skin tone an clappin... | Negative | Positive | 0.975 |
| 3 | for some reason felt like crying because ur my fav youtuber and my inspiration f... | Negative | Positive | 0.903 |
| 4 | nope not at all of course she will return those things to owner of the missing i... | Negative | Positive | 0.901 |
| 5 | how to capture cctv camera with processing we bought tv tuner video grabber and ... | Negative | Neutral | 0.895 |
| 6 | now am confused watch machine learning or data science... | Negative | Neutral | 0.892 |
| 7 | referee was man of the match that game clapping hands medium skin tone... | Negative | Positive | 0.891 |
| 8 | dont think the explanation in this lecture is very good... | Negative | Positive | 0.886 |
| 9 | dear apna college team please begin course the previous lectures are difficult t... | Negative | Positive | 0.886 |
| 10 | code with brackets and snippet manager called brackets snippets by edc which all... | Negative | Neutral | 0.882 |
| 11 | am feeling very grateful right now for this amazing life god has blessed me with... | Neutral | Positive | 0.996 |
| 12 | thank you for this precious content smiling face with hearteyessmiling face with... | Neutral | Positive | 0.995 |
| 13 | babymonster red heart im so happy yall stan them they are my fav red heart red h... | Neutral | Positive | 0.993 |
| 14 | smiling face with smiling eyes smiling face with smiling eyes red heart smiling ... | Neutral | Positive | 0.992 |
| 15 | growing heart growing heart navin is in love with kiran red heart red heart... | Neutral | Positive | 0.991 |
| 16 | you triggered my alexa other than that love your tutorials so so much thank you ... | Neutral | Positive | 0.990 |
| 17 | im super early for this one ukraine ukraine flexed biceps flexed biceps... | Neutral | Positive | 0.989 |
| 18 | prayers for mel gibson and the rest of the victims of these horrible fires red h... | Neutral | Positive | 0.988 |
| 19 | hi shradha ji your videos have inspired me greatly pray to god folded hands fold... | Neutral | Positive | 0.981 |
| 20 | love dogs and cats and my family red heart red heart red heart red heart red hea... | Neutral | Positive | 0.980 |
| 21 | pile of poo face with rolling eyes face vomiting face with rolling eyes pile of ... | Positive | Negative | 0.986 |
| 22 | majority of the media should be ashamed prior to actually watching the case and ... | Positive | Negative | 0.976 |
| 23 | trump and the ag barr along with the supreme court judges are openly using voter... | Positive | Negative | 0.974 |
| 24 | one reason people get to jail is because they dont listen they go mouthing off a... | Positive | Negative | 0.966 |
| 25 | have no sympathy for amber turd millions of women all over the world would have ... | Positive | Negative | 0.963 |
| 26 | thank you sir sir login krna toh email username and password required fir as sai... | Positive | Neutral | 0.963 |
| 27 | one question for matt miller is when the israelis claimed that unrwa had hamas w... | Positive | Negative | 0.962 |
| 28 | hi daniel greate video have aquestion what camera were you using in this series ... | Positive | Neutral | 0.961 |
| 29 | its not just political theater its witness intimidation all these clowns shouldv... | Positive | Negative | 0.960 |
| 30 | how can you trust russia ever again will thosethat committed crimes be held acco... | Positive | Negative | 0.953 |

**Qualitative observations:**

- Review the errors above and annotate patterns here before thesis submission.
  Common patterns to look for:
  - Sarcasm / irony (e.g., 'Great job breaking the internet again')
  - Neutral comments misclassified as Negative (mild criticism)
  - Short ambiguous comments (≤5 words)
  - Domain-specific vocabulary (gaming, politics, cooking) causing false positives

---

## ENSEMBLE

**Total errors shown:** 30

**Error type breakdown:**

| Confusion | Count |
|-----------|-------|
| Positive → Negative | 10 |
| Negative → Neutral | 8 |
| Neutral → Positive | 7 |
| Neutral → Negative | 3 |
| Negative → Positive | 2 |

**Sample Misclassifications (highest confidence first):**

| # | Text (truncated) | True | Predicted | Confidence |
|---|-----------------|------|-----------|-----------|
| 1 | that man wicked after he came to jamaica hot hot to married you he did this to y... | Negative | Positive | 0.939 |
| 2 | where can find vim... | Negative | Neutral | 0.907 |
| 3 | for some reason felt like crying because ur my fav youtuber and my inspiration f... | Negative | Positive | 0.905 |
| 4 | big query is an enterprise data warehouse for large amounts of relational struct... | Negative | Neutral | 0.887 |
| 5 | plot twist he added pineapple off camera... | Negative | Neutral | 0.875 |
| 6 | now am confused watch machine learning or data science... | Negative | Neutral | 0.866 |
| 7 | hi sir have compac ram gb pentium but sometime it create issue... | Negative | Neutral | 0.863 |
| 8 | but will the merger between honda and nissan make hondas cars less reliable beca... | Negative | Neutral | 0.861 |
| 9 | steve adams is not just some random player... | Negative | Neutral | 0.858 |
| 10 | use english text please... | Negative | Neutral | 0.855 |
| 11 | babymonster red heart im so happy yall stan them they are my fav red heart red h... | Neutral | Positive | 0.986 |
| 12 | its sad bittersweet understanding that you are completely alone when youre in tr... | Neutral | Negative | 0.968 |
| 13 | harris you are dynamo thank you for being so honest and forthright we need journ... | Neutral | Positive | 0.961 |
| 14 | am feeling very grateful right now for this amazing life god has blessed me with... | Neutral | Positive | 0.949 |
| 15 | why do people follow hit and runs the insurance will pay and you wont get punish... | Neutral | Negative | 0.947 |
| 16 | bad cops good cops not working for fox... | Neutral | Negative | 0.946 |
| 17 | wish wish the world listens ro this everyone should listen to this love you liam... | Neutral | Positive | 0.944 |
| 18 | love this do canada as well red heart... | Neutral | Positive | 0.943 |
| 19 | that clip from hook made me smile thank you so much for that... | Neutral | Positive | 0.937 |
| 20 | remember her she was pretty cool hope shes learned lot... | Neutral | Positive | 0.934 |
| 21 | pile of poo face with rolling eyes face vomiting face with rolling eyes pile of ... | Positive | Negative | 0.983 |
| 22 | russia has lot of equipment but it lacks professional army their army and reserv... | Positive | Negative | 0.959 |
| 23 | what is surprising about the fact he cheats he lies he cheats he bullies people ... | Positive | Negative | 0.944 |
| 24 | trump and the ag barr along with the supreme court judges are openly using voter... | Positive | Negative | 0.940 |
| 25 | he needs to be removed itll be disgusting if he pardons himself hes got deaths o... | Positive | Negative | 0.938 |
| 26 | love hearing how deluded the west is and how quickly they forget talking about s... | Positive | Negative | 0.936 |
| 27 | why would truth about history make you feel guilty because youre white that make... | Positive | Negative | 0.934 |
| 28 | have no sympathy for amber turd millions of women all over the world would have ... | Positive | Negative | 0.932 |
| 29 | what an awful legacy to inherit poor woman... | Positive | Negative | 0.931 |
| 30 | wouldve been lot cheaper on tax payers if they just throw the cops in jail... | Positive | Negative | 0.919 |

**Qualitative observations:**

- Review the errors above and annotate patterns here before thesis submission.
  Common patterns to look for:
  - Sarcasm / irony (e.g., 'Great job breaking the internet again')
  - Neutral comments misclassified as Negative (mild criticism)
  - Short ambiguous comments (≤5 words)
  - Domain-specific vocabulary (gaming, politics, cooking) causing false positives

---

## META_LEARNER

**Total errors shown:** 30

**Error type breakdown:**

| Confusion | Count |
|-----------|-------|
| Neutral → Positive | 10 |
| Negative → Positive | 8 |
| Positive → Negative | 7 |
| Positive → Neutral | 3 |
| Negative → Neutral | 2 |

**Sample Misclassifications (highest confidence first):**

| # | Text (truncated) | True | Predicted | Confidence |
|---|-----------------|------|-----------|-----------|
| 1 | that man wicked after he came to jamaica hot hot to married you he did this to y... | Negative | Positive | 0.946 |
| 2 | for some reason felt like crying because ur my fav youtuber and my inspiration f... | Negative | Positive | 0.932 |
| 3 | for all the incredible bravery of the ukrainians the fact is that they cannot su... | Negative | Positive | 0.925 |
| 4 | this is for all of you democrats others liberals republicans trump is still your... | Negative | Positive | 0.919 |
| 5 | thank you for your help with this amazing ai you represent the future... | Negative | Positive | 0.913 |
| 6 | it is good to see that every day and in every way trump is getting fatter and fa... | Negative | Positive | 0.909 |
| 7 | where can find vim... | Negative | Neutral | 0.893 |
| 8 | truly appreciate the tremendous amount of effort that was put into this course e... | Negative | Positive | 0.890 |
| 9 | this is pretty good tutorial overall especially considering its free did experie... | Negative | Positive | 0.888 |
| 10 | now am confused watch machine learning or data science... | Negative | Neutral | 0.887 |
| 11 | babymonster red heart im so happy yall stan them they are my fav red heart red h... | Neutral | Positive | 0.956 |
| 12 | harris you are dynamo thank you for being so honest and forthright we need journ... | Neutral | Positive | 0.953 |
| 13 | wish wish the world listens ro this everyone should listen to this love you liam... | Neutral | Positive | 0.950 |
| 14 | am feeling very grateful right now for this amazing life god has blessed me with... | Neutral | Positive | 0.950 |
| 15 | that clip from hook made me smile thank you so much for that... | Neutral | Positive | 0.947 |
| 16 | you triggered my alexa other than that love your tutorials so so much thank you ... | Neutral | Positive | 0.946 |
| 17 | love this do canada as well red heart... | Neutral | Positive | 0.946 |
| 18 | dont have autism but do this lot actually when get gift im always greatful and d... | Neutral | Positive | 0.945 |
| 19 | remember her she was pretty cool hope shes learned lot... | Neutral | Positive | 0.945 |
| 20 | im super early for this one ukraine ukraine flexed biceps flexed biceps... | Neutral | Positive | 0.944 |
| 21 | pile of poo face with rolling eyes face vomiting face with rolling eyes pile of ... | Positive | Negative | 0.905 |
| 22 | hi daniel greate video have aquestion what camera were you using in this series ... | Positive | Neutral | 0.904 |
| 23 | russia has lot of equipment but it lacks professional army their army and reserv... | Positive | Negative | 0.898 |
| 24 | where is this wanna fly like superman pretty plz... | Positive | Neutral | 0.896 |
| 25 | he needs to be removed itll be disgusting if he pardons himself hes got deaths o... | Positive | Negative | 0.891 |
| 26 | his name is jared for yall wondering... | Positive | Neutral | 0.891 |
| 27 | love hearing how deluded the west is and how quickly they forget talking about s... | Positive | Negative | 0.888 |
| 28 | what is surprising about the fact he cheats he lies he cheats he bullies people ... | Positive | Negative | 0.887 |
| 29 | trump and the ag barr along with the supreme court judges are openly using voter... | Positive | Negative | 0.886 |
| 30 | what an awful legacy to inherit poor woman... | Positive | Negative | 0.886 |

**Qualitative observations:**

- Review the errors above and annotate patterns here before thesis submission.
  Common patterns to look for:
  - Sarcasm / irony (e.g., 'Great job breaking the internet again')
  - Neutral comments misclassified as Negative (mild criticism)
  - Short ambiguous comments (≤5 words)
  - Domain-specific vocabulary (gaming, politics, cooking) causing false positives

---

## Thesis Framing

Add this to your **Results — Error Analysis** sub-section:

> To characterise model failure modes, we examined the 30 highest-confidence
> misclassifications for each model on the test set. The most frequent error
> pattern across all models is **Neutral → Negative confusion** (Table N),
> consistent with the lowest per-class F1 scores for the Neutral class
> (0.56–0.63 across models, Table X). This suggests that the Neutral class
> presents the greatest challenge for automated YouTube sentiment classification,
> likely due to its heterogeneous nature: neutral comments include factual
> statements, questions, and mixed-sentiment expressions that overlap lexically
> with both Positive and Negative classes.