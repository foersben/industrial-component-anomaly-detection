---
title: "Notes"
type: "notes"
---
* final report on "industrial component anomaly detection" project
* look at our initial report first in docs/.report/misc and create a latex document using the same writing style - we are beginners in this line of work so keep it plain/simple, well explained
* then checkout our dummy implementations and the baseline/patchcore implementation in this project and describe (or make up) our thinking process, how we initially progressed within the project
* mention the initial setup of a real software architecture instead of notebooks using fastapi and streamlit added preprocessing steps (clahe and gaussian blur)
* afterwards checkout the notebook notebooks/shared/03_modelling_autoencoder.ipynb and describe our working and thinking process (learning process) and add information that we also used testwise a patch strategy to feed overlapping patches to the initial auto encoder concepts in the notebooks (not sure I think those notebooks are not inside the project at the moment) - so in the narrative simply describe the architectures of the initial notebooks that I referenced for our convolutional autoencoders
* further the masked image thing was already applied test wise in the notebooks
* in the notebooks we used at the beginning only mse (or i think it was mse) for the loss
* then in the subsequent notebook we tested ssim in isolation
* only afterwards in our keras model I added the combined approach of ssim in combination with mse according to Bergmann
* afterwards you can simply start checking the commits and the progress we made on this branch on our keras cae which is the most advanced model with all its details as describe in the /docs zensical documentation and all its details
* for evaluation we want to focus on f1 score (image and pixel level) and aupimo pixel level performance, the loss functions, the pr-curve and threshold curve with aupimo
* we will use an image resolution of 256*256, we also decided for our final report to use for evaluation a train/val split of 85 + 15 = 100 (whole train set) the test set stays untouched during training
* finally, I added optuna for an optimised hyperparameter search instead of randomly guessing and checking the outcome - make sure to describe optuna and why we use it
* we are currently computing the hyperparameter
* those are then used to compute the final performance evaluation of our models and add the final evaluation to the report

* please for now add placeholders for the final evaluation that we can easily replace later
