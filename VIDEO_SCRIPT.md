# Demo video script (target 6 to 7 minutes)

**0:00 Problem (30s).** Risk models give a bare percentage. Clinicians want to see *where* disease may be. Show the title and the disclaimer banner.

**0:30 Data and models (60s).** Z-Alizadeh Sani extension, 303 patients. Four targets. LAD, LCX, RCA and Cath are removed from inputs to prevent leakage. Show `results/metrics.md` and the ROC plot. Say honestly that CAD and LAD are strong and LCX and RCA are weaker.

**1:30 Live demo (2.5 min).**
1. Open the app. Median patient: all arteries green.
2. Click **High-risk**. Arteries animate to orange and red, and the CAD card reads "CAD likely".
3. Change one input (age, then typical chest pain). Show the risk updating live.
4. Rotate and zoom. Use the Front, Back, Left, Right and Top buttons.
5. Click the LAD. Show the tooltip, the detail card and the LAD SHAP drivers.
6. Switch tabs (Overall CAD, LAD, LCX, RCA) and explain the red and blue bars and the share percentages.
7. Open **Model performance** and point out the table.

**4:00 Technical walkthrough (2 min).** Architecture diagram: Excel -> `train.py` -> joblib -> FastAPI (`/predict` returns probabilities and SHAP) -> Three.js. Show `features.py` as the single schema, the one colour function shared by the 3D scene and the dashboard, the model registry, and the real BodyParts3D RCA branches plus procedural vessels.

**6:00 Limitations and extensions (45s).** Small single-centre dataset. Procedural LAD and LCX. Add real meshes through `manifest.json`. Not a medical device.
