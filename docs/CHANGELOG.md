# Change Loh

## 1.0 June 20, 2026
- Added Autocriteria Script, logging mechanics, and a skeleton for the standalone app

## 1.1 June 26, 2026
- Frontend now working, added output files, QtCore for logging, and better run functionality

## 1.1.1 July 5, 2026
- Moved frontend to tkinter for stability
- Moved openai_client as legacy code. multiple provider support framework is added, just need the api keys now
- Loading now appears in a progress bar, it is defaulted at number of trials now but will be changed to actual progress once the project is near complete

## 1.1.2 July 8, 2026
- Add skeleton for anomaly detection and validation. The anomaly detector will rely first on deterministic checks and return a review flag with severity and explanation. The validation agent will work trial by trial and look at original eligibility text vs extracted criteria and look for innacuracies. We will manually annotate 10 trials so we can look at recall and precision
- This will be made a standalone later