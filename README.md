Based on Code sourced from; [/Dymensium/PointNuker](https://github.com/Dymensium/PointNuker)

With PointNuker-SOR, the Statistical Outlier Removal (SOR) filter is a precision tool used to "denoise" your 3D Gaussian Splats by identifying points that are too far from their neighbors compared to the average density of the cloud. 

# **SOR Settings**

SOR typically uses two primary parameters to determine what stays and what gets "nuked": 

For every point in the cloud, the algorithm identifies its nearest neighbours (where k is a user-defined parameter) and computes the average distance from the point to those neighbours

**Neighbours (k):**

Function: Defines the size of the neighborhood to analyze for each point. The tool calculates the average distance from a point to its nearest neighbors. Higher values are slower
Adjustment: Increase this for denser splats to get a more reliable average; decrease it if the filter is accidentally selecting fine details like thin wires or hair. 
Allowable Range: 1 to 500

**Std Ratio (nSigma) - Standard Deviation Multiplier Threshold:**

Function: Sets the sensitivity threshold for deletion. Any point whose mean distance to its neighbors is greater than the global mean distance plus times the standard deviation is removed.
Adjustment: Lower values (e.g., 0.5–0.8) are more aggressive and will remove more floaters; higher values (e.g., 2.0–3.0) are more conservative, keeping points unless they are extremely isolated.
Allowable Range: 0.01 to 10

**Presets**
Conservative:  k=30  & nSigma= 2.0
Balanced:      k=20  & nSigma= 1.5
Aggressive:    k=10  & nSigma= 1.0

[[Introduction Video](https://www.youtube.com/watch?v=F9r-f9q6Bds)]

<img width="2127" height="1404" alt="SOR" src="https://github.com/user-attachments/assets/ec5b5689-f478-46a0-af0f-2b5a2cbb0c8e" />

SOR - [Statistical  Outlier Removal](https://www.google.com/search?sourceid=chrome&aep=42&source=chrome.crn.rb&udm=50&q=SOR+-+Statistical+Outlier+Removal&mstk=AUtExfCaZiZh6kMy4e6evvU3lqDW4R3FYdttLDTlanyoAMulrRJhh9AqQSSl9jfsLJSzNHBCa8I1cKZxxg29LpBky17W-a0kx1ORmBKypXPY6_sTIWx1aIAn1c2HsU5QYGsJ-39OPSlmXzdsnFwTzuTOSqWI6qgOHOiBepAN8z2F3Wmfna0oeGX60Lpbmy2hsGd626Db-_ERzqGVTMWtI6-ahLJ1PYIcNXXE2vx7bqv0WOtW9Z81EACduLZGLs2PM-fhbExsoXEPGZ7L-trVEIVlQiMVKObm-98aayM&csuir=1&mtid=KKblaavcKY2KnesPwdWb8Qc)  - There two parts to the Plugin:   

1) Run SOR on an existing 3dGS PLY file  - the code is currently 'attached' to the bottom of  the "Rendering"  tab.  I have coded the process to ONLY split: so you get to Isolate and inspect before deleting. This means you can crop etc on the outliers & then merge the model-splits back into one single model.  When 'merging' you can enter the name you want in the input box.  If you have multiple models in the Scene you can 'isolate' only those you want to work with simply by changing the visibility - SOR will only process the visible ones.  I'm not sure what the max count capacity is.

<img width="1014" height="842" alt="image" src="https://github.com/user-attachments/assets/7fda29b3-6ffd-4568-8d66-84d3de8c8ce8" />

2A) COLMAP Points3D.txt/bin editor: This is appended to the bottom of the "Training" Tab and allows pre-processing of the bin/txt point files - there are no graphics . Culling uses the same SOR as above & cropping is on an AABB system or sphere.  Before cropping you can interrogate the file (sphere sample given) - This doesn't prefill in the inputs though. After crop or crop/sor it will save to bin/txt file & a bak file is made.

<img width="680" height="464" alt="image" src="https://github.com/user-attachments/assets/5cb5a1f8-282c-4082-b68f-0aa2a2d574d8" />

2B) [Edit COLMAP (GUI editor)](https://github.com/bgofish/SOR_Plugin/wiki)  This is appended to the bottom of the "Training" Tab (Upper) - Allows for scale/rotation/translation & cropping but no SOR.

<img width="2559" height="1599" alt="image" src="https://github.com/user-attachments/assets/1bf31a0f-c6ef-49bd-a9e2-95149c9e41d3" />
