# Author: Monika Dörig, 2023
# Questions & Inputs to: monika.doerig@balgrist.ch




from nipype.interfaces.base import (CommandLine, traits, TraitedSpec,
                                    BaseInterface, BaseInterfaceInputSpec, File)
import os


class NiftiNordicInputSpec(BaseInterfaceInputSpec):
    additional_path = traits.Str(desc='Additional path to be added with addpath in MATLAB', mandatory=True)
    fn_magn_in = traits.File(exists=True, desc='Input magnitude file', mandatory=True)
    fn_phase_in = traits.File(exists=True, desc='Input phase file', mandatory=True)
    fn_out = traits.Str(desc='Output prefix or suffix', mandatory=True)
    ARG = traits.File(exists=True, desc='Path to MATLAB struct file representing input parameters')
                      

class NiftiNordicOutputSpec(TraitedSpec):
    # traits.Either allows for the possibility that the file might not exist (None) or that it exists and is a valid file (File(exists=True))
    output_file = traits.Either(None, File(exists=True), desc='Denoised nii file')
    add_info_file = traits.Either(None, File(exists=True), desc='Optional output file for additional info')
    complex_phase_file = traits.Either(None, File(exists=True), desc='Optional output file - saves phase in similar format as input phase')
    complex_magn_file = traits.Either(None, File(exists=True), desc='Optional output file - saves magn in similar format as input magn')
    gfactor_map_file = traits.Either(None, File(exists=True), desc='Optional output file for gfactor map')
    residual_mat_file = traits.Either(None, File(exists=True), desc='Optional output file for NORDIC residuals')

class NiftiNordic(BaseInterface):
    input_spec = NiftiNordicInputSpec
    output_spec = NiftiNordicOutputSpec
    
    def _run_interface(self, runtime):
        # Third argument for NIFTI_NORDIC_nipype.m may need adaptation depending on the the suffix of your file: here it is defined for a .nii.gz file --> [0:-7]
        mlab_cmd = f"matlab -r \"addpath('{self.inputs.additional_path}'); try; NIFTI_NORDIC_nipype ('{self.inputs.fn_magn_in}', '{self.inputs.fn_phase_in}', '{os.path.basename(self.inputs.fn_magn_in)[0:-7]}{self.inputs.fn_out}','{self.inputs.ARG}' ); catch ME; disp(getReport(ME)); end; quit;\""
        mlab = CommandLine(command=mlab_cmd, terminal_output='stream', resource_monitor=True)
        result = mlab.run()  
        return result.runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        
        # Check if the output files exist (depending on the arguments) and assign outputs 
        output = os.path.abspath(f"{os.getcwd()}/{os.path.basename(self.inputs.fn_magn_in)[0:-7]}{self.inputs.fn_out}.nii")
        if os.path.exists(output):
            outputs['output_file'] = output
            
        info = os.path.abspath(f"{os.getcwd()}/{os.path.basename(self.inputs.fn_magn_in)[0:-7]}{self.inputs.fn_out}_info.mat")
        if os.path.exists(info):
            outputs['add_info_file'] = info
            
        complex_phase = os.path.abspath(f"{os.getcwd()}/{os.path.basename(self.inputs.fn_magn_in)[0:-7]}{self.inputs.fn_out}_phase.nii")
        if os.path.exists(complex_phase):
            outputs['complex_phase_file'] = complex_phase
            
        complex_magn = os.path.abspath(f"{os.getcwd()}/{os.path.basename(self.inputs.fn_magn_in)[0:-7]}{self.inputs.fn_out}_magn.nii")
        if os.path.exists(complex_magn):
            outputs['complex_magn_file'] = complex_magn
            
        gfactor = os.path.abspath(f"{os.getcwd()}/gfactor_{os.path.basename(self.inputs.fn_magn_in)[0:-7]}{self.inputs.fn_out}.nii")
        if os.path.exists(gfactor):
            outputs['gfactor_map_file'] = gfactor     
            
        residuals = os.path.abspath(f"{os.getcwd()}/RESIDUAL{os.path.basename(self.inputs.fn_magn_in)[0:-7]}{self.inputs.fn_out}.mat")
        if os.path.exists(residuals):
            outputs['residual_mat_file'] = residuals
            
        return outputs