#

🪨3DA: Rapid Drilling Data Visualisation and Analysis for Geologists
Ever stared at endless spreadsheets of drilling data wondering where to start and how to make sense of it all? As geologists, we spend countless hours trying to understand our deposits, constantly returning to our data with new questions and hypotheses. The challenge? Most software packages require significant data formatting before you can even begin your analysis. This is where 3DA steps in - a purpose-built tool that transforms drilling data into meaningful visualisations in minutes, not hours. Built with Python and Streamlit, 3DA streamlines the process that many geologists know too well: taking raw drilling data and turning it into actionable insights. As a resource geologist with a keen interest in data science, I developed 3DA to bridge the gap between raw data and geological understanding, automating the tedious parts so you can focus on interpretation.
Where Does 3DA Fit In?
Think of 3DA as your first port of call - a powerful tool for quick visualisation and initial analysis. It directly handles typical csv files and the standard drilling data formats supplied to the Australian State Geological Surveys, letting you get straight to the good stuff. Once you've identified interesting patterns or areas that need more investigation, you might want to move your data into IoGAS for more powerful multivariate statistical analysis or Leapfrog for detailed 3D modelling.
The Power of Exploratory Data Analysis
Exploratory Data Analysis (EDA) is an iterative process in understanding and modelling geology. The beauty of 3DA is that it removes the leg work needed to get that initial view of your data. No more wating time formatting files just to see if there's something worth investigating further. Simply spin the model around, apply some filters, check the statistics, and decide where to focus your detailed analysis.

---

3DA in Action: A Practical Overview
The 3DA Interface
3DA features an intuitive tab-based interface that guides you through the data analysis workflow:
Data Loading: Upload and prepare your drilling data with support for standard CSV formats and Australian State Geological Survey formats
Visualisations: Explore your data in interactive 3D space with customizable views, filters, and exaggeration options
Statistics: Analyze distributions, correlations, and summary statistics across your dataset
Clustering: Apply geochemical clustering techniques to identify domains and patterns
Export Data: Save your processed data for use in other applications or reporting

This simple navigation structure allows you to move freely between different analysis types while maintaining the context of your dataset.
1. Getting Started - Data Upload & Compositing
First things first - getting your data in. 3DA directly handles the data files in typical csv format with the headers in the first row or the formats specified by the Australian State Geological Surveys with the headers in the H1000 row. Once you have your files, the upload interface is straightforward and the with three main sections:
Collar Data: Contains your drillhole locations and orientations
Assay Data: Your sample elemental concentrations
Lithology Data: Downhole logging and logging code dictionary

Column mapping and element selection.
Once your data is loaded, 3DA's compositing tool offers flexible compositing options that let you standardise your sampling intervals using length-weighted, averaging methods. This flexibility allows you to view and analyse your data at whatever resolution best suits your needs. Compositing standardises your data, smoothing out noise while preserving important trends, and can make cross-hole comparisons more reliable. The tool's immediate visual feedback helps you quickly optimise your composite parameters to match your deposit's characteristics and your analytical objectives.
3DA Compositing Options

2. The 3D Visualisation
At the heart of 3DA is its interactive 3D viewer, designed specifically for drilling data. Once loaded, your data springs to life in a fully manipulatable 3D space where you can:
Examine drillholes in true 3D space
Offset grade and lithology traces for clearer viewing of both datasets simultaneously
Rotate, zoom, and pan with intuitive mouse controls
Adjust vertical exaggeration to enhance subtle features in shallow drilling

3D drillhole visualisation
Powerful Filtering Options:
Select specific drillholes or drilling fences
Filter intervals by lithology or alteration codes
Set grade thresholds
Toggle between grade views, lithology views, or combined displays

Combined (Grade and Lithology) drillhole view filtered on a selected drillhole.
Navigation is straightforward, left-click and drag to rotate, right-click and drag to pan, and use your mouse wheel to zoom. The grade filter slider lets you dynamically adjust minimum and maximum values, making it easy to identify and investigate high-grade zones and their relationships to specific rock types.
Pro Tips:
Start by filtering to a single drill fence to get familiar with your geology
Use the combined grade-lithology view to spot relationships between mineralisation and rock types
Experiment with vertical exaggeration to better visualise shallow or deep features

3. Summary Statistics - Understanding Your Data
3DA provides a statistical dashboard that gives you an immediate overview of your dataset:
Statistical summaries showing means, quartiles, and potential outliers
Histograms for all numeric fields
One-click transformation between linear and log scales  - particularly useful for elements like gold or arsenic that often show strong right-skew distributions. 

Statistical dashboard showing grade distributions and summary statistics.4. Swath Analysis - The Big Picture
One of 3DA's most powerful features is its interactive swath analysis. These plots reveal how grades vary across your deposit through:
East-West and North-South grade trends along customisable sections
Vertical grade changes with depth
Sample density overlays to validate data coverage
Grade variability bands showing standard deviation - super helpful for understanding grade consistency

The tool automatically calculates and displays these trends, helping you identify grade patterns, structural controls, and potential domains. You can adjust the number of bins and section widths to optimize the analysis for your deposit scale.

Swath plots showing grade trends across the deposit with standard deviation bands (red) and sample density overlay.
5. Correlation Analysis
Understanding element relationships is crucial for geochemical interpretation. The correlation tools help identify element associations that might indicate mineralisation styles or alteration patterns. 3DA's correlation tools help you explore these relationships through:
Interactive correlation matrices
Scatter plots for element pairs
Identify element associations

Global correlation matrix.
Selected scatter plots and correlation matrix 

6. Logged Unit Analysis
Lithologies and logged units play a crucial role in domain definition and deposit understanding. While multivariate geochemistry analysis often helps refine geological domain boundaries, well-logged lithology and alteration can provide initial constraints for domaining. 3DA provides several tools for analysing your logged units:
Box plots for each rock type
Grade statistics by lithology
Interactive filtering
Minimum sample thresholds

These built-in visualisations help you quickly understand element distributions between rock types. The boxplots effectively display where statistical populations differ, highlighting potential domain boundaries that align with logged units.
Summary statistics by logged units
Box plots displaying elemental concentration for each lithology.
7. Finding the Good Stuff
3DA includes a flexible significant intervals calculator that lets you quickly identify and report your best drill intersections. You can:
Set custom grade cutoffs for different elements
Specify minimum intersection lengths
Adjust maximum internal waste tolerance
View results with associated lithologies
Export results as formatted tables for reporting

Customisable significant intervals calculator with adjustable cutoffs and internal waste parameters
8. Geochemical Clustering - Quick Domain Insights
One of the most powerful features in 3DA is the geochemical clustering tool. While detailed geological logging is valuable, it often faces challenges:
Inconsistency between geologists
Variable logging quality
Subtle mineralogical changes not visible to the naked eye
Time-consuming manual review of large datasets

Geochemical clustering provides a rapid, objective first pass at identifying major rock types and alteration domains. 3DA offers two approaches: straightforward K-means clustering of your raw data, or a more advanced option that first transforms the data using Principal Component Analysis (PCA) before clustering. Both methods have their merits - direct clustering is simpler and more transparent, while PCA transformation can help reduce noise and highlight subtle patterns. 
To demonstrate the core workflow, we'll use the simpler K-means approach on a dataset where the geology consists of a weathered profile of transported cover, saprolite and saprock.
Remember, while clustering is powerful, it should always be integrated with geological understanding and validated against known geology, structure and controls on mineralisation.
Feature Selection: Before clustering, it's important to select appropriate elements. Using 3DA's biplot and correlation matrix visualisations, we can identify key elements while removing those that are redundant or noisy. As geochemical data typically shows skewed distributions, we apply a natural log transform to normalise the data.
Geochemical clustering feature selection and optionsPro Tips:
Start with major elements (Al, Fe, Ca, Mg, Na, K) for broad lithological domains
Add pathfinder elements to refine alteration and mineralisation patterns

Clustering: Clustering: Applying k-means clustering to the transformed geochemical data revealed three distinct domains. While the weathering profile contains additional complexity, this three-cluster solution provides a simple test case, capturing the main lithological units of interest. The choice is supported by both the scree plot's elbow at k=3 and the coherent spatial groupings that form in the 3D viewer
K-means clustering scree plot3D PCA of Clusters along the first 3 principal components

Statistical Validation: Summary statistics and box plots reveal distinct geochemical fingerprints for each cluster:
Cluster 0 (Saprock): Elevated base metals
Cluster 1(Lower saprolite): Lower concentrations of mobile elements (Na, Mg, and Ca) compared to cluster 0
Cluster 2 (Cover): Depleted in base metals and mobile elements

Box plots displaying the distributions of the selected featuresComparison with Logged Data: 3DA's lithology vs cluster heatmap provides a quick correlate clustering results against traditional logging. For example, the heatmap allows us to cross check our clusters against logged geology: Cluster 0 correlates strongly with the saprock (rssr) as well as the minor andesite (vanv) and monzonite (icmo) units, Cluster 1 correlates with the upper (rssu) and lower (rssl) saprolite units, and Cluster 2 correlates most strongly with the logged gravel (ssgr) units.
Cluster vs logged units heat map

Spatial Validation: The critical final step is checking that clusters form coherent geological shapes in 3D space - random or scattered patterns would indicate poor clustering results. In our case, these distinct geochemical signatures generally align well with the logged weathering boundaries. The clusters display strong horizontal continuity, reflecting the expected geometry of weathering horizons. 
3D validation of created geochemical clustersWhen viewed in 3D alongside the lithological logging, the geochemical domains show better hole-to-hole correlation than the original logging, particularly in the southeastern portion of the project. Here, two holes show significant logging inconsistencies compared to the broader dataset - a common challenge when multiple geologists are involved in logging weathered material. 
3D validation of geochemical clusters offset against logged unitsWhile visual logging can effectively identify these weathering horizons, determining the exact boundaries between lower saprolite, upper saprolite, and saprock units often remains subjective. The objective nature of geochemical clustering helps standardise these boundary determinations. If we wanted to delve into further detail, applying k-means with k=4 would reveal distinct geochemical differences between the northern and southern saprolite units, reflecting variations in weathering intensity across the deposit.

3D visualisation of geochemical clusters with increased resolution (k=4)

Save Processed Data: Once you're happy with your clusters, you can export the processed data for use in other applications - for example, importing into Leapfrog to create 3D geological models guided by the geochemical domains. These clusters will also serve as our "ground truth" domains in the next article, where we'll explore how machine learning can use these geochemical patterns to predict domains in new drilling data.

