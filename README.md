![crypto1](https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/d7a3c903-907c-410a-9c5b-89c28a0e6fec)

----

# **Cryptocurrency Market Segmentation: K-Means Clustering With and Without Principal Component Analysis**

----

## **Project Overview**

This project applies unsupervised machine learning to investigate whether short-term price movements — specifically 24-hour and 7-day percentage changes — can meaningfully segment a universe of 42 cryptocurrencies into distinct behavioral clusters. The core analytical question is not merely whether K-Means clustering works on this data, but whether Principal Component Analysis (PCA) dimensionality reduction improves the quality, clarity, and interpretability of the resulting clusters compared to clustering on the original feature space. The project demonstrates proficiency across the full unsupervised learning pipeline: data normalization, optimal cluster number determination using four independent validation methods, dimensionality reduction, clustering, and multi-dimensional visualization.

---

## **Methodology**

All cryptocurrency price change data was first normalized using scikit-learn's `StandardScaler` function, ensuring that no single variable disproportionately influences the clustering algorithm due to scale differences. The optimal number of clusters K was then determined independently for both the original scaled data and the PCA-reduced data using four complementary validation methods: Within-Cluster Sum of Squares (WCSS) Elbow, Calinski-Harabasz, Silhouette, and Davies-Bouldin. Each method approaches the cluster quality question from a different mathematical perspective, and their convergence — or divergence — provides a robust and multi-faceted basis for the final K selection.

PCA was then applied to reduce the original feature space to three principal components, and the entire K determination and clustering process was repeated on the reduced data to enable direct comparison.

---

### *K Selection: Original Data (Figure 2.2.1)*

<img width="979" alt="crypto2" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/9f1f582f-9c0e-4ee0-8442-a683e3a71e3f">

The four validation method plots for the original scaled data present a nuanced but ultimately convergent picture. The WCSS Elbow curve descends steeply from K=2 through K=4, after which the rate of decline flattens noticeably — the characteristic elbow shape that indicates diminishing returns from adding more clusters. The dashed line at K=4 correctly identifies this inflection point.

The Calinski-Harabasz score, which measures the ratio of between-cluster dispersion to within-cluster dispersion (higher is better), peaks sharply at K=4 before leveling off, providing strong independent confirmation of K=4 as the optimal choice. The Silhouette score — which measures how well each point fits its assigned cluster relative to neighboring clusters, with higher values indicating better-defined clusters — peaks at K=3 with a value of approximately 0.70, then drops sharply at K=4 to approximately 0.32 before declining further. This divergence between Silhouette and the other metrics is worth noting: the Silhouette score suggests K=3 may produce more internally cohesive clusters, while WCSS and Calinski-Harabasz favor K=4. The Davies-Bouldin score (lower is better, as it measures average cluster similarity) reaches its minimum at K=3 with a value near 0.18, before rising at K=4 and fluctuating through higher K values. Taken together, the four methods suggest K=4 as the primary recommendation — supported by two of the four metrics — with K=3 as a viable alternative worth examining.

---

### *K-Means Clustering: Original Data (Figures 3.3.1, 3.4.1, 3.4.3, 3.4.4)*

<img width="948" alt="crypto3" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/2bf20568-1d60-427d-b3fc-198704f68324">

<img width="939" alt="crypto4" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/922e02da-1275-40ce-ab52-afa09677d180">

<img width="928" alt="crypto5" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/019804c5-0f2a-4f7b-861b-f15a6c35ba5d">

<img width="919" alt="crypto6" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/e55ba3a6-4251-460f-9c43-435472082e40">

The 2D scatter plots in Figure 3.3.1 visualize K-Means clustering across K=2, K=3, and K=4 using 24-hour and 7-day price change percentages as axes. At K=4, the data resolves into a recognizable structure: a large, dispersed yellow cluster occupying the central negative-to-neutral 7-day return zone; a tighter red cluster of coins with positive 7-day returns; a compact isolated blue cluster far to the left representing a single outlier cryptocurrency with an extreme negative 24-hour return of approximately -6%; and a smaller orange cluster of moderate negative returners. The two K=4 plots are nearly identical, confirming clustering stability.

At K=3, the yellow and orange clusters merge into one large grouping, losing the distinction between moderate and significant negative returners. At K=2, virtually all coins collapse into a single blue cluster, with only the extreme outlier segregated — a configuration that destroys almost all meaningful structure.

The 3D scatter plots (Figures 3.4.1, 3.4.3, 3.4.4) add the 14-day price change as a third dimension, revealing additional structural depth. At K=4 (Figure 3.4.1), four clusters are clearly visible in three-dimensional space: two central overlapping clusters of the majority of coins, an isolated yellow outlier cluster in the upper-left representing a single high-positive-14-day-return coin, and an orange cluster of moderate-return coins offset to the right. At K=3 (Figure 3.4.3), the outlier cluster remains isolated but the central mass loses differentiation. At K=2 (Figure 3.4.4), the 3D space reveals the extreme limitation of binary clustering — one enormous cluster dominates while the outlier sits alone. The 3D visualizations confirm that K=4 captures meaningful structure that K=2 and K=3 cannot resolve.

---

### *K Selection: PCA Data (Figure 5.2.1)*

<img width="987" alt="crypto7" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/dbe54819-dfa2-46ac-8224-994197c51f2b">

Applying PCA to reduce the feature space to three principal components and repeating the K determination process produces a notably different set of validation curves. The WCSS Elbow plot retains its elbow at K=4, with the inflection point now more pronounced and the curve descending more steeply overall — suggesting that PCA has sharpened the data's natural cluster structure by removing noise. The Silhouette score improves substantially in the PCA space: at K=3 it reaches approximately 0.75 (higher than the 0.70 in the original space), and at K=4 it falls to approximately 0.42 — a less severe drop than the 0.32 seen without PCA, indicating better cluster cohesion at K=4 in the reduced space. The Davies-Bouldin score reaches its minimum at K=3 (approximately 0.14) and remains lower across all K values compared to the original data, confirming improved cluster separation. The Calinski-Harabasz score, however, behaves very differently in PCA space — it rises continuously from K=2 through K=10 without a clear peak, suggesting that the PCA transformation has altered the between-cluster dispersion structure in a way that this metric cannot resolve cleanly. This is the one metric that fails to recommend K=4 with PCA data, and it is the outlier in an otherwise consistent picture.

Balancing all four metrics, K=4 remains the recommended optimal configuration in PCA space — confirmed by WCSS, partially supported by Silhouette, and consistent with the visual cluster analysis.

---

### *K-Means Clustering: PCA Data (Figures 6.3.1–6.3.3, 6.4.1–6.4.3)*


<img width="988" alt="crypto8" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/2b1a8d83-ae1a-48f6-b171-45778697c4ff">

<img width="988" alt="crypto9" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/278df51d-5511-4480-8a19-c73907ce11dc">

<img width="967" alt="crypto10" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/943a4770-6a1b-41d1-80cb-c49efe94f21b">


<img width="935" alt="crypto11" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/cac4b894-28fd-438a-8896-8ff5bf2cfcb4">

<img width="951" alt="crypto12" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/a0207cd5-b581-4f00-852d-e28a5c0de660">

<img width="969" alt="crypto13" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/5af3b439-be80-48a0-900b-feadc7e240e0">

<img width="959" alt="crypto14" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/c0da1e44-8a0f-44c0-9c8e-54d7d34ef1c2">

The 2D PCA scatter plots in Figures 6.3.1 through 6.3.3 plot all pairwise combinations of the three principal components (PCA1 vs. PCA2, PCA1 vs. PCA3, PCA2 vs. PCA3) for K=2, K=3, K=4, and K=10. The improvement in cluster quality relative to the original data is immediately visually apparent across all three projection planes.

In the PCA1 vs. PCA2 plots (Figure 6.3.1), the K=4 configuration shows four well-separated clusters: a large yellow majority cluster near the origin, a compact red cluster of high-positive-return coins slightly offset, a distant orange singleton in the upper-right representing a strong outlier, and a well-separated blue singleton in the lower-right. The clusters are more spatially distinct and better separated than their counterparts in the original feature space. At K=10 (upper-right panel), the proliferation of overlapping cluster boundaries in the central region vividly illustrates the danger of over-clustering — many of the ten clusters share members and overlap substantially, confirming that K=10 fragments rather than discovers meaningful structure. The PCA1 vs. PCA3 and PCA2 vs. PCA3 projections (Figures 6.3.2 and 6.3.3) reinforce these observations from different geometric perspectives, each revealing the same fundamental four-cluster structure from a different angle.

The 3D PCA scatter plots (Figures 6.4.1 through 6.4.3 and Image 13) provide the clearest and most compelling visualization of the advantages PCA confers. At K=4 (Figure 6.4.1 / Image 10), the four clusters are dramatically more spatially separated in PCA space than in the original feature space: the large yellow and red clusters occupy the central region with clear boundaries between them, while the blue and orange singleton clusters are positioned at distant, unambiguous locations in the PCA coordinate system. The cluster boundaries are clean, the within-cluster groupings are tight, and the between-cluster separations are large — all hallmarks of high-quality clustering. At K=10 (Image 11), the central cluster mass fragments into numerous overlapping groups, visually confirming that additional clusters beyond four add noise rather than signal. At K=3 (Image 12), the singleton clusters remain well-separated but the central mass loses meaningful internal differentiation. At K=2 (Image 13), the structure collapses into one dominant cluster and one near-empty one — the PCA space makes this failure mode even more visually obvious than the original feature space did.

---

### *Direct Comparison: Original vs. PCA (Figures 7.1.1, 7.1.2, 7.2.1, 7.2.2, 7.3.1)*

<img width="520" alt="crypto15" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/e7d3af85-0fa8-4bce-938e-0c726fa75604">

<img width="497" alt="crypto16" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/2b0e3c3d-f80f-4ce4-8798-15cbfe602132">


<img width="487" alt="crypto17" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/0bb0a6d0-8a55-4347-8566-bdfe25f85d12">

<img width="502" alt="crypto18" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/b03b404f-09f7-4841-8673-7b12854afc7d">


<img width="957" alt="crypto19" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/ec68f842-c838-40da-b8dc-9478360f539f">

<img width="973" alt="crypto20" src="https://github.com/njgeorge000158/Cryptocurrency-Cluster-Analysis-with-PCA-Using-Scikit-Learn/assets/137228821/4be55a34-b9e9-4b0d-ba09-7ce527c30455">

The side-by-side WCSS Elbow comparisons (Figures 7.1.1 and 7.1.2) reveal that the PCA elbow curve is both lower in absolute WCSS values and exhibits a sharper, more pronounced inflection at K=4 — indicating that PCA has reduced within-cluster variance and made the optimal K determination more unambiguous. The starting WCSS value at K=2 drops from approximately 197 in the original space to approximately 165 in PCA space, and the curve flattens more decisively after K=4.

The 2D scatter plot comparison (Figures 7.2.1 and 7.2.2) makes the clustering improvement concrete and visual. In the original space (Figure 7.2.1), the K=4 clusters show significant overlap between the yellow and orange groups near the origin, and the cluster boundaries are difficult to distinguish. In the PCA space (Figure 7.2.2), the same four clusters are dramatically more separated: the blue singleton occupies a distant lower-right position (PCA1 ≈ 7, PCA2 ≈ -4), the orange singleton sits in the upper center (PCA1 ≈ 5, PCA2 ≈ 7), and the yellow and red majority clusters, while adjacent, show cleaner internal cohesion and more distinct centroids. The visual improvement is unambiguous.

The 3D comparison (Figures 7.3.1 original and PCA) completes the picture. The original 3D plot shows four clusters but with considerable overlap and ambiguity in the central region. The PCA 3D plot shows the same four clusters but with notably tighter groupings and wider inter-cluster distances — the defining characteristics of superior clustering quality.

---

## **The Advantages of PCA: A Direct Answer**

The evidence across all 19 visualizations converges on a clear and consistent answer to the central question. PCA confers four specific and measurable advantages in this analysis:

**Noise reduction.** By compressing the original feature space into three principal components that capture the maximum variance, PCA filters out low-signal dimensions that add noise to the clustering process without contributing meaningful structure. The result is tighter within-cluster groupings and more distinct between-cluster boundaries.

**Sharper K determination.** The WCSS Elbow inflection at K=4 is more pronounced in PCA space, and the Silhouette scores are higher across all K values, making the optimal cluster number easier to identify with greater confidence.

**Improved cluster separability.** Across every 2D and 3D visualization, PCA-clustered data shows greater spatial separation between clusters than the original data. The singleton outlier clusters in particular are dramatically more isolated in PCA space, making them easier to identify and interpret.

**Dimensionality reduction without information loss.** Three PCA components successfully preserve the essential structure of the original multi-dimensional feature space while making that structure more accessible to the K-Means algorithm and more interpretable in visualization — demonstrating that, for this dataset, much of the original feature information was redundant or noisy rather than analytically valuable.

---

## **Conclusion**

The optimal configuration for clustering this cryptocurrency dataset is K=4, confirmed consistently across both the original and PCA-reduced feature spaces. Four clusters effectively segment the 42 cryptocurrencies into: a large majority group of stable, low-volatility coins; a group of positive short-term performers; and two singleton clusters representing individual coins with extreme or anomalous price behavior. PCA dimensionality reduction does not change the optimal K or the fundamental cluster structure — but it makes that structure cleaner, tighter, more visually interpretable, and more analytically defensible. For cryptocurrency market analysis, these improvements translate directly into more reliable segmentation and more trustworthy insights.

----

### Copyright

Nicholas J. George © 2023. All Rights Reserved.
