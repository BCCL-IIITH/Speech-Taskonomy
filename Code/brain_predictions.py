#!/usr/bin/env python
# coding: utf-8

# This cell imports libraries that you will need
# Run this.
from matplotlib.pyplot import figure, cm
import os
import numpy as np
import utils1
import h5py
from sklearn.model_selection import KFold
import logging
logging.basicConfig(level=logging.DEBUG)


# # Load Speech features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = int)
    parser.add_argument("layers", help="Choose layers", type = int)
    parser.add_argument("featurename", help="Choose feature", type = str)
    parser.add_argument("outputdir", help="Choose layers", type = str)
    args = parser.parse_args()

    num_layers = args.layers

    wav_features = np.load('./taskonomy_features/'+args.featurename, allow_pickle=True)
    print(wav_features.shape)

    pieman_features = wav_features.item()['pieman']
    print(pieman_features[0]).shape


    # # Remove Starting: 14 and ending: 10 TRs (Silence of starting 21secs and ending 15 secs)

    from npp import zscore
    Rstim = []
    for eachbatch in np.arange(num_layers):
        Rstim.append(np.vstack([np.array(pieman_features[eachbatch][10:-8])]))

    # Print the sizes of these matrices
    print ("Rstim shape: ", Rstim[0].shape)


    # ### Concatenate delayed stimuli for FIR model
    # Next you are going to concatenate multiple delayed versions of the stimuli, in order to create a linear [finite impulse response (FIR) model](http://en.wikipedia.org/wiki/Fir_filter). This is a vitally important step, and is conceptually a bit difficult, so take a few minutes to make sure you understand what is going on here.
    # 
    # #### Background: the hemodynamic response
    # First you need to understand the problem that the FIR model is solving. fMRI measures the blood-oxygen level dependent (BOLD) signal, which is a complicated and nonlinear combination of blood oxygenation and blood volume. When neurons in an area of the brain become active, they start using up lots of energy. To compensate, nearby blood vessels dilate so that more oxygen and glucose become available to the neurons. The resulting changes in blood oxygenation (which increases) and volume (which also increases) create the magnetic signature that is recorded by fMRI. 
    # 
    # But this process is **slow**. It takes seconds after the neural activity begins for the blood vessels to dilate and for the BOLD response to become apparent. And then it takes more seconds for the response to go away. So although a neural response might only last milliseconds, the associated BOLD response will rise and fall over a span of maybe 10 seconds, orders of magnitude slower. The shape of this rise and fall is called the [hemodynamic response function (HRF)](http://en.wikipedia.org/wiki/Haemodynamic_response).
    # 
    # Here is a pretty standard looking example of an HRF:
    # 
    # <img src='http://www.brainmatters.nl/wp-content/uploads/bold.png' width=350px></img>
    # 
    # #### FIR model
    # To accurately model how the brain responds to these stimuli we must also model the HRF. There are many ways to do this. The most common is to assume that the HRF follows a canonical shape. But this approach turns out to not work very well: different parts of the brain have very different vasculature (blood vessels), so the HRF shape can vary a lot. 
    # 
    # Instead, what you are going to here is estimate a separate HRF for each semantic feature in each voxel that is being modeled. This estimate is going to take the form of a linear finite impulse response (FIR) model. The linear FIR form is particularly nice to use because it's very simple to estimate and powerful (if anything, it might be too powerful.. more on that later). To build a linear FIR model all you have to do is concatenate together multiple delayed copies of the stimulus. I usually use four delays: 1, 2, 3, and 4 time points. The resulting delayed features can be thought of as representing the stimulus 1, 2, 3, and 4 time points ago. So the regression weights for those features will represent how a particular voxel responds to a feature 1, 2, 3, or 4 time points in the past, and these regression weights are a 4-point estimate of the HRF for that feature in that voxel.
    # 
    # The potential downside of the FIR model is that it may be too expressive. Each feature in each voxel is allowed to have any HRF, but this comes at the cost of multiplying the total number of regression weights that we must fit by the number of delays. In all likelihood the true HRFs vary, but they don't vary that much, so we probably don't need this many independent features. This cost becomes apparent if you increase the number of delays. This will slow down model fitting and likely decrease the stability of the regression weights, leading to decreased model performance. 
    # 
    # Feel free to play around with the number of delays and see how it affects the model results!



    # Delay stimuli
    from util import make_delayed
    ndelays = 8
    delays = range(1, ndelays+1)

    print ("FIR model delays: ", delays)

    delRstim = []
    for eachbatch in np.arange(num_layers):
        delRstim.append(make_delayed(Rstim[eachbatch], delays))


    # Print the sizes of these matrices
    print ("delRstim shape: ", delRstim[0].shape)

    # ## Response data
    # Next you will load the fMRI data. This is totally the most exciting part! These responses have already been preprocessed (the 3D images were motion corrected and aligned to each other, detrended, and then z-scored within each stimulus) so you don't have to worry about that.
    # 
    # You will load three different variables: `zRresp`, the responses to the regression dataset; `zPresp`, the responses to the prediction dataset; and `mask`, which is a 3D mask showing which voxels have been selected (we are not modeling every voxel in the scan, that would take forever, we are only modeling the voxels that overlap with the cerebral cortex).


    # Load training data for subject 1, reading dataset 
    #roi_voxels = np.load('../../afni-nosmooth/pieman_sub_L_avg.npy',allow_pickle=True)
    roi_voxels = np.load('../../afni-nosmooth/pieman_sub_R_avg.npy',allow_pickle=True)
    roi_voxels = roi_voxels[10:-8,:]
    print(roi_voxels.shape)

    from npp import zscore
    zRresp = []
    for eachsubj in np.arange(roi_voxels.shape[0]):
        zRresp.append(roi_voxels[eachsubj])
    zRresp = np.array(zRresp)


    # Print matrix shapes
    print ("zRresp shape (num time points, num voxels): ", zRresp.shape)


    # ## Regression model
    # Finally, the core of the analysis: you will fit a regression model that predicts the responses of each voxel as a weighted sum of the semantic features. This model will then be tested using a held out dataset (the Prediction dataset). And if the model proves to be reasonably predictive, then the weights of the regression model will tell us what semantic features each voxel responds to.
    # 
    # This is a linear regression model, so if the response time course for voxel $j$ is $R_j$, the stimulus time course for semantic feature $i$ is $S_i$, and the regression weight for feature $i$ in voxel $j$ is $\beta_{ij}$, then the model can be written as:
    # 
    # $$\hat{R}_j = \beta_{0j} S_0 + \beta_{1j} S_1 + \cdots$$
    # 
    # or:
    # 
    # $$\hat{R}_j = \sum_i \beta_{ij} S_i$$
    # 
    # The trick, of course, is accurately estimating the $\beta_j$ values. This is commonly done by minimizing the sum of the squared error (here across time, $t$):
    # 
    # $$E_j(\beta) = \sum_t (R_{jt} - \hat{R}_{jt})^2 = \sum_t (R_{jt} - \sum_i \beta_{i} S_{it})^2$$
    # 
    # $$\beta_j = \underset{\beta}{\operatorname{argmin}} E_j(\beta)$$
    # 
    # Computing $\beta$ this way is called ordinary least squares (OLS), and this will not work in our case because the total number of features (3940) is smaller than the number of time points (3737). (It would be possible if the number of delays was smaller than 4, but it would give terrible results.. feel free to try it! OLS can be performed using the function `np.linalg.lstsq`.)
    # 
    # In almost every case, linear regression can be improved by making some prior assumptions about the weights (or, equivalently, about the covariance structure of the stimuli). This is called **regularization**, or **regularized linear regression**. One way to do this is to penalize the error function by the sum of the squared weights. This is commonly known as **ridge regression**, and is a special case of [Tikhonov regularization](http://en.wikipedia.org/wiki/Ridge_regression). It finds the $\beta$ that minimizes the following error function:
    # 
    # $$E_j(\beta) = \sum_t (R_{jt} - \sum_i \beta_{i} S_{it})^2 + \alpha \sum_i \beta_i^2$$
    # 
    # (In practice we will use a different formulation that involves re-weighting the singular values of the matrix $S$ before computing its pseudoinverse. This method achieves the same results but is extremely efficient because it uses all the linear algebra machinery that computers are so good at to build many models in parallel.)
    # 
    # ### The hyperparameter: $\alpha$
    # You may have noticed in the equation above that we have introduced a new parameter, $\alpha$, which controls the strength of the regularization. If $\alpha$ is set to zero, then we get back to exactly the OLS formulation (above). As $\alpha$ goes to infinity, the regularization forces all the weights to go to zero (in practice this also has the slightly weirder effect of making all the weights independent, as if each feature was regressed separately on the responses).
    # 
    # So how do we choose $\alpha$? We're going to do it here using cross-validation. First, we split the Regression dataset up into two parts. Then we estimate the weights for a given $\alpha$ on the first part, and test how well we can predict responses on the second part. This is repeated for each possible $\alpha$ that we want to test, and for a couple different splits of the Regression dataset. Then we find the $\alpha^*$ that gave us the best predictions within the split Regression dataset. Finally we estimate the weights using the entire Regression dataset and the selected $\alpha^*$.
    # 
    # Because this is an annoying and laborious process, I've encapsulated it within the function `bootstrap_ridge`. You simply give this function your datasets, the possible $\alpha$ values, and a few parameters for the cross-validation, and it does all the rest. The parameter `nboots` determines the number of cross-validation tests that will be run. 
    # 
    # To do cross-validation, `bootstrap_ridge` divides the Regression dataset into many small chunks, and then splits those chunks into the two groups that will be used to estimate weights and test $\alpha$ values. This is better than just choosing individual time points because both the fMRI data and stimuli are autocorrelated (i.e. correlated across time). The parameter `chunklen` determines the length of the chunks, and the parameter `nchunks` determines the number of chunks in the $\alpha$-testing dataset. 
    # 
    # Running the regression will take a few minutes.


    # Run regression
    from ridge import bootstrap_ridge
    from scipy import stats
    #alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    nboots = 2 # Number of cross-validation runs.
    chunklen = 40 # 
    nchunks = 20
    kf = KFold(n_splits=4)
    save_dir = 'taskonomy_predictions/'+args.outputdir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for eachbatch in np.arange(num_layers):
        save_dir = str(eachbatch)
        if not os.path.exists('taskonomy_predictions/'+args.outputdir+'/'+save_dir):
            os.mkdir('taskonomy_predictions/'+args.outputdir+'/'+save_dir)
        for run in np.arange(1):
            count = 0
            all_preds = []
            all_reals = []
            all_corrs = []
            for train_index, test_index in kf.split(zRresp):
                    alphas = np.logspace(1, 3, 10)
                    # remove 5 TRs which either precede or follow the TRs in the test set

                    train_index_without_overlap = train_index
                    for rem_val in range(test_index[0] - 5, test_index[0], 1):
                        train_index_without_overlap = train_index_without_overlap[train_index_without_overlap != rem_val]

                    for rem_val in range(test_index[-1] + 1, test_index[-1] + 6, 1):
                        train_index_without_overlap = train_index_without_overlap[train_index_without_overlap != rem_val]

                    x_train, x_test = delRstim[eachbatch][train_index_without_overlap], delRstim[eachbatch][test_index]
                    y_train, y_test = zRresp[train_index_without_overlap], zRresp[test_index]
                    
                    x_train = stats.zscore(x_train,axis=0)
                    x_train = np.nan_to_num(x_train)

                    x_test = stats.zscore(x_test,axis=0)
                    x_test = np.nan_to_num(x_test)

                    y_train = stats.zscore(y_train,axis=0)
                    y_train = np.nan_to_num(y_train)

                    y_test = stats.zscore(y_test,axis=0) 
                    y_test = np.nan_to_num(y_test)
                    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
                    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(x_train, y_train, x_test, y_test,
                                                                         alphas, nboots, chunklen, nchunks,
                                                                         singcutoff=1e-10, single_alpha=True)
                    y_pred = np.dot(x_test, wt)
                    print ("pred has shape: ", y_pred.shape)
                    #np.save(os.path.join('Treedepth/'+save_dir, "y_pred_{}".format(count)),y_pred)
                    #np.save(os.path.join('Treedepth/'+save_dir, "y_test_{}".format(count)),y_test)
                    all_reals.append(y_test)
                    all_preds.append(y_pred)
                    all_corrs.append(corr)

                    count+=1
            all_reals = np.vstack(all_reals)
            all_preds = np.vstack(all_preds)
            all_corr = np.vstack(all_corrs)

            voxcorrs = np.zeros((all_reals.shape[1],)) # create zero-filled array to hold correlations
            for vi in range(all_reals.shape[1]):
                voxcorrs[vi] = np.corrcoef(all_reals[:,vi], all_preds[:,vi])[0,1]
            print (voxcorrs)
            print(np.mean(voxcorrs[np.where(voxcorrs>0.0)[0]]))

            np.save(os.path.join('taskonomy_predictions/'+args.outputdir+'/'+save_dir, "rhavg4_layer_"+str(eachbatch)+'_run'+str(run)),np.mean(all_corr,axis=0))
            np.save(os.path.join('taskonomy_predictions/'+args.outputdir+'/'+save_dir, "rhavg5_layer_"+str(eachbatch)+'_run'+str(run)),voxcorrs)