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

## 1.2 July 21, 2026
- On the way to implement a matching algorithm so the dictionary autoupdates. The idea is to do Human-In-the-loop continual terminology normalization. We will implement an embedding-based semantic matching algorithm, and then feed it into a validation agent that decides whether to accept, reject, or request review. There will then be a human feedback loop that stores accepted mappings, and a continual learning dictionary will be generated from there. This ends by having a trained normalization model once we have a large corpus of terms, then we can likely take a reinforcement learning approach to scale
- added a normalization workflow that utilizes `sentence-transformers` and added test scripts for semantic matching
- decluttered matching dictionaries
