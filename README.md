# NORDIC_Raw
Matlab code for performing image reconstruction in MRI and performing the NORDIC denoising

The two files NORDIC and NIFTI_NORDIC perform similar concepts, locally low-rank denoising.
Both approaches, uses a g-factor map to flatten the noise, and a noise-scan for estimating the homoegenous noise.
For NORDIC, the noise-scan and the g-factor are explicit constructions provided as the last elements in a 4D array.
For NIFTI_NORDIC, these are estimated based on the data. The construction for estimating the g-factor noise and the thermal noise level
uses the MPPCA method of Veraart et al. 2016
NIFTI_NORDIC has additional paramters that can be adjusted, for learning or understanding the influence of the different algortimic choices.
For NIFTI_NORDIC, there are two different options, depending on whether dMRI or fMRI is used. 
This difference appears related to the hwo the phase is retained in the DICOM of the vendor software. A corresponding distinction is not neccesary for the NORDIC processing.

This version of NIFTI_NORDIC has been made possible through the testing and evaulation of many people, including


Logan Dowdle
Luca Vizioli
Cheryl Olman
Essa Yacoub
Henry Braun
Remi Patriat
Federico De Martino
Lonike Faes
Torben Ellegaard Lund
Lasse Knudsen
Stamatios Sotiropoulos
Karen Mullinger
Daniel Marsh
Susan Francis
Jose Manzano Patron


Any questions, ciomments or suggestions can be directed to

Steen Moeller
moell018@umn.edu
