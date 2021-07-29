# NORDIC_Raw
Matlab code for performing image reconstruction in MRI and performing the NORDIC denoising


# Overview
The two files NORDIC and NIFTI_NORDIC perform similar concepts, locally low-rank denoising.
Both approaches, uses a g-factor map to flatten the noise, and a noise-scan for estimating the homoegenous noise.
For NORDIC, the noise-scan and the g-factor are explicit constructions provided as the last elements in a 4D array.
For NIFTI_NORDIC, these are estimated based on the data. The construction for estimating the g-factor noise and the thermal noise level
uses the MPPCA method of Veraart et al. 2016
NIFTI_NORDIC has additional paramters that can be adjusted, for learning or understanding the influence of the different algortimic choices.
For NIFTI_NORDIC, there are two different options, depending on whether dMRI or fMRI is used. 
This difference appears related to the hwo the phase is retained in the DICOM of the vendor software. A corresponding distinction is not neccesary for the NORDIC processing.

This version of NIFTI_NORDIC has been made possible through the testing and evaulation of many people, including


Logan Dowdle,
Luca Vizioli,
Cheryl Olman,
Essa Yacoub,
Henry Braun,
Remi Patriat,
Mehmet Akcakaya,
Federico De Martino,
Lonike Faes,
Torben Ellegaard Lund,
Lasse Knudsen,
Stamatios Sotiropoulos,
Karen Mullinger,
Daniel Marsh,
Susan Francis,
Jose Manzano Patron


Any questions, ciomments or suggestions can be directed to

Steen Moeller
moell018@umn.edu

# System Requirements
# Hardware Requirements
Package only requires a standard computer with enough RAM to support the in-memory operations and loading the data
# Software Requirements
 This package is tested on Matlab version 2017b. All neccesary dependencies are part of the default matlab installation
# Installation Guide
 Ensure that NORDIC.m is in a path that is visible to matlab
# Demo for the installation
   Using the NORDIC.m function and the simulation in DEMO, the following will demonstrate hwo to use NORDIC

    script_for_creating_simulation_data
    NORDIC('demo_data_for_NORDIC.mat')
    
    QQ=load('KSP_demo_data_for_NORDICkernel8')
    Q=load('demo_data_for_NORDIC') 
    figure; clf
    subplot(2,2,1); imagesc(sq(real(Q.KSP(:,:,32,12))),[0 1]); title('Data + noise')
    subplot(2,2,2); imagesc(sq(real(Q.IMG(:,:,32,12))),[0 1]); title('Data w/o noise')
    subplot(2,2,3); imagesc(sq(real(QQ.KSP_update(:,:,32,12))),[0 1]); title('NORDIC processed')
    subplot(2,2,4); plot(sq(real(Q.KSP(20,25,32,1:end-2)  -   Q.IMG(20,25,32,1:end-1)))), hold on
                    plot(sq(real(QQ.KSP_update(20,25,32,1:end)  -   Q.IMG(20,25,32,1:end-1))))
                    legend('difference before NORDIC','difference after NORDIC')

 



